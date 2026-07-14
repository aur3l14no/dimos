// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dimos_module::{error_throttled, run_with_transport, warn_throttled, Input, Module, Output};
use dimos_voxel_ray_tracing::voxel_ray_tracer::{
    batch_local_bounds, emit_points, update_map, Config, LocalBounds, VoxelKey, VoxelMap,
};
use lcm_msgs::geometry_msgs::{Point, Pose, PoseStamped, Quaternion};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{PointCloud2, PointField};
use lcm_msgs::std_msgs::{Header, Time};
use nalgebra::{UnitQuaternion, Vector3};
use tokio::sync::Notify;

struct PoseSample {
    stamp: f64,
    pose: MatchedPose,
}

#[derive(Clone)]
struct MatchedPose {
    translation: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
    parent_frame: String,
    child_frame: String,
}

struct MapJob {
    cloud: PointCloud2,
    pose: MatchedPose,
}

/// A replaceable slot keeps only the newest value and wakes its single consumer.
struct LatestSlot<T> {
    pending: Mutex<Option<T>>,
    wake: Notify,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self {
            pending: Mutex::new(None),
            wake: Notify::new(),
        }
    }
}

impl<T> LatestSlot<T> {
    fn replace(&self, value: T) {
        *self.pending.lock().expect("latest slot mutex") = Some(value);
        self.wake.notify_one();
    }

    async fn next(&self) -> T {
        loop {
            self.wake.notified().await;
            if let Some(value) = self.pending.lock().expect("latest slot mutex").take() {
                return value;
            }
        }
    }
}

#[derive(Module)]
#[module(setup = spawn_worker, teardown = stop_worker)]
struct RayTracingVoxelMap {
    #[input(decode = PointCloud2::decode, handler = on_lidar)]
    lidar: Input<PointCloud2>,

    #[input(decode = Odometry::decode, handler = on_odometry)]
    odometry: Input<Odometry>,

    #[output(encode = PointCloud2::encode)]
    global_map: Output<PointCloud2>,

    #[output(encode = PointCloud2::encode)]
    local_map: Output<PointCloud2>,

    // Cylinder bounds of the local map. Position is the center, orientation holds
    // radius, z_min, z_max. Stamped like local_map so consumers pair them.
    #[output(encode = PoseStamped::encode)]
    region_bounds: Output<PoseStamped>,

    #[config]
    config: Config,

    poses: VecDeque<PoseSample>,
    // Latest cloud waiting for an exact pose or a later-pose watermark.
    pending_cloud: Option<PointCloud2>,
    latest_job: Arc<LatestSlot<MapJob>>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl RayTracingVoxelMap {
    async fn spawn_worker(&mut self) {
        let worker = MapWorker {
            latest_job: Arc::clone(&self.latest_job),
            config: self.config.clone(),
            global_map: self.global_map.clone(),
            local_map: self.local_map.clone(),
            region_bounds: self.region_bounds.clone(),
        };
        self.worker = Some(tokio::spawn(worker.run()));
    }

    async fn stop_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            handle.abort();
            handle_worker_exit(handle.await, true);
        }
    }

    async fn ensure_worker_running(&mut self) {
        // The derive runtime cannot select a module-owned task. Until it supports
        // supervised tasks, continuous sensor input propagates a worker panic on
        // the next message; teardown still aborts and joins the worker.
        let Some(handle) = self.worker.as_ref() else {
            panic!("map worker is not running");
        };
        if !handle.is_finished() {
            return;
        }
        let handle = self.worker.take().expect("worker checked above");
        handle_worker_exit(handle.await, false);
    }

    async fn on_odometry(&mut self, msg: Odometry) {
        self.ensure_worker_running().await;
        let p = &msg.pose.pose.position;
        let q = &msg.pose.pose.orientation;
        let reset = push_pose(
            &mut self.poses,
            PoseSample {
                stamp: time_secs(&msg.header.stamp),
                pose: MatchedPose {
                    translation: Vector3::new(p.x as f32, p.y as f32, p.z as f32),
                    rotation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        q.w as f32, q.x as f32, q.y as f32, q.z as f32,
                    )),
                    parent_frame: msg.header.frame_id.clone(),
                    child_frame: msg.child_frame_id.clone(),
                },
            },
        );
        if reset {
            self.pending_cloud = None;
            warn_throttled!(
                Duration::from_secs(1),
                "Odometry frame or timestamp epoch changed; cleared pending pose/cloud join state.",
            );
        }
        self.try_hand_off_cloud();
    }

    async fn on_lidar(&mut self, msg: PointCloud2) {
        self.ensure_worker_running().await;
        self.pending_cloud = Some(msg);
        self.try_hand_off_cloud();
    }

    fn try_hand_off_cloud(&mut self) {
        let Some(cloud) = self.pending_cloud.as_ref() else {
            return;
        };
        let pose = match match_pose_for_cloud(&self.poses, time_secs(&cloud.header.stamp)) {
            CloudPoseMatch::Pending => return,
            CloudPoseMatch::Miss => {
                warn_throttled!(
                    Duration::from_secs(1),
                    "No odometry within tolerance after crossing the cloud stamp; dropped a cloud.",
                );
                self.pending_cloud = None;
                return;
            }
            CloudPoseMatch::Matched(pose) => pose,
        };

        if !cloud.header.frame_id.is_empty() && cloud.header.frame_id != pose.child_frame {
            warn_throttled!(
                Duration::from_secs(1),
                cloud_frame = %cloud.header.frame_id,
                odometry_child_frame = %pose.child_frame,
                "Cloud frame does not match odometry child frame; dropped a cloud.",
            );
            self.pending_cloud = None;
            return;
        }

        let cloud = self.pending_cloud.take().expect("cloud checked above");
        self.latest_job.replace(MapJob { cloud, pose });
    }
}

fn handle_worker_exit(result: Result<(), tokio::task::JoinError>, cancellation_expected: bool) {
    match result {
        Err(error) if cancellation_expected && error.is_cancelled() => {}
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => panic!("map worker stopped unexpectedly: {error}"),
        Ok(()) => panic!("map worker exited unexpectedly"),
    }
}

#[derive(Default)]
struct MapState {
    map: VoxelMap,
    frame_count: u32,
    batch_points: Vec<(f32, f32, f32)>,
    batch_origins: Vec<(f32, f32, f32)>,
    frame_id: Option<String>,
    last_cloud_stamp: Option<f64>,
}

enum CloudEpoch {
    Current,
    New,
}

impl MapState {
    /// Accept a strictly newer cloud in the current frame. A changed frame or
    /// clear timestamp rewind starts a new map epoch; duplicate and slightly
    /// out-of-order clouds are dropped without disturbing the current map.
    fn prepare_for_cloud(&mut self, frame_id: &str, stamp: f64) -> Option<CloudEpoch> {
        if frame_id.is_empty() {
            warn_throttled!(
                Duration::from_secs(1),
                "Odometry parent frame is empty; dropped a cloud.",
            );
            return None;
        }

        let mut epoch = CloudEpoch::Current;
        if self
            .frame_id
            .as_deref()
            .is_some_and(|current| current != frame_id)
        {
            warn_throttled!(
                Duration::from_secs(1),
                previous_frame = ?self.frame_id,
                new_frame = %frame_id,
                "Odometry parent frame changed; starting a new voxel-map epoch.",
            );
            *self = Self::default();
            epoch = CloudEpoch::New;
        } else if self
            .last_cloud_stamp
            .is_some_and(|last| last - stamp > POSE_MATCH_TOLERANCE_S)
        {
            warn_throttled!(
                Duration::from_secs(1),
                previous_stamp = ?self.last_cloud_stamp,
                new_stamp = stamp,
                "Cloud timestamp rewound; starting a new voxel-map epoch.",
            );
            *self = Self::default();
            epoch = CloudEpoch::New;
        } else if self.last_cloud_stamp.is_some_and(|last| stamp <= last) {
            warn_throttled!(
                Duration::from_secs(1),
                previous_stamp = ?self.last_cloud_stamp,
                new_stamp = stamp,
                "Duplicate or out-of-order cloud timestamp; dropped a cloud.",
            );
            return None;
        }

        self.frame_id = Some(frame_id.to_string());
        self.last_cloud_stamp = Some(stamp);
        Some(epoch)
    }
}

struct EmissionPlan {
    bounds: Option<PoseStamped>,
    local_bounds: Option<LocalBounds>,
    global_due: bool,
    stamp: Time,
    frame_id: String,
    live: ahash::AHashSet<VoxelKey>,
}

struct MapWorker {
    latest_job: Arc<LatestSlot<MapJob>>,
    config: Config,
    global_map: Output<PointCloud2>,
    local_map: Output<PointCloud2>,
    region_bounds: Output<PoseStamped>,
}

impl MapWorker {
    async fn run(self) {
        let mut state = MapState::default();
        loop {
            let job = self.latest_job.next().await;
            self.process(&mut state, job).await;
        }
    }

    async fn process(&self, state: &mut MapState, job: MapJob) {
        let Some(plan) = tokio::task::block_in_place(|| self.update(state, job)) else {
            return;
        };
        let EmissionPlan {
            bounds,
            local_bounds,
            global_due,
            stamp,
            frame_id,
            live,
        } = plan;

        if let Some(bounds) = bounds {
            if let Err(error) = self.region_bounds.publish(&bounds).await {
                error_throttled!(
                    Duration::from_secs(1),
                    error = %error,
                    "Region bounds failed to publish",
                );
            }
        }
        if global_due {
            let global = tokio::task::block_in_place(|| {
                let points = emit_points(&state.map, self.config.voxel_size, None, 0, &live);
                points_to_cloud(&points, &frame_id, stamp.clone())
            });
            publish_cloud(&self.global_map, &global).await;
        }
        if let Some(local_bounds) = local_bounds {
            let local = tokio::task::block_in_place(|| {
                let points = emit_points(
                    &state.map,
                    self.config.voxel_size,
                    Some(&local_bounds),
                    self.config.support_min,
                    &live,
                );
                points_to_cloud(&points, &frame_id, stamp)
            });
            publish_cloud(&self.local_map, &local).await;
        }
    }

    fn update(&self, state: &mut MapState, job: MapJob) -> Option<EmissionPlan> {
        let MapJob { cloud, pose } = job;
        let translation = pose.translation;
        let origin = (translation.x, translation.y, translation.z);
        let voxel_size = self.config.voxel_size;

        let points = match extract_xyz(&cloud) {
            Ok(p) => p,
            Err(e) => {
                warn_throttled!(
                    Duration::from_secs(1),
                    error = %e,
                    "Failed to get lidar points, dropped a cloud.",
                );
                return None;
            }
        };
        if points.is_empty() {
            return None;
        }

        let out_frame_id = pose.parent_frame.as_str();
        let new_epoch = match state.prepare_for_cloud(out_frame_id, time_secs(&cloud.header.stamp))
        {
            Some(CloudEpoch::Current) => false,
            Some(CloudEpoch::New) => true,
            None => return None,
        };

        // Transform sensor-frame points into the odometry parent frame.
        let rot = pose.rotation.to_rotation_matrix();
        let points: Vec<(f32, f32, f32)> = points
            .iter()
            .map(|&(x, y, z)| {
                let p = rot * Vector3::new(x, y, z) + translation;
                (p.x, p.y, p.z)
            })
            .collect();

        let live = update_map(&mut state.map, origin, &points, &self.config);

        // The batch only feeds the local region bounds, so skip it when the local
        // map is disabled.
        if self.config.emit_every > 0 {
            state.batch_points.extend_from_slice(&points);
            state.batch_origins.push(origin);
        }

        state.frame_count += 1;
        let local_due = emit_due(state.frame_count, self.config.emit_every, new_epoch);

        let (bounds, cylinder) = if local_due {
            let margin = self.config.shadow_depth + voxel_size;
            let (cx, cy, radius, z_min, z_max) = batch_local_bounds(
                &state.batch_points,
                &state.batch_origins,
                self.config.region_percentile,
                margin,
            );
            state.batch_points.clear();
            state.batch_origins.clear();

            let bounds_msg = PoseStamped {
                header: Header {
                    seq: 0,
                    stamp: cloud.header.stamp.clone(),
                    frame_id: out_frame_id.to_string(),
                },
                pose: Pose {
                    position: Point {
                        x: cx as f64,
                        y: cy as f64,
                        z: 0.0,
                    },
                    orientation: Quaternion {
                        x: radius as f64,
                        y: z_min as f64,
                        z: z_max as f64,
                        w: 0.0,
                    },
                },
            };
            (
                Some(bounds_msg),
                Some(LocalBounds {
                    origin_x: cx,
                    origin_y: cy,
                    r_xy_max_sq: radius * radius,
                    z_min,
                    z_max,
                }),
            )
        } else {
            (None, None)
        };

        Some(EmissionPlan {
            bounds,
            local_bounds: cylinder,
            global_due: emit_due(state.frame_count, self.config.global_emit_every, new_epoch),
            stamp: cloud.header.stamp,
            frame_id: out_frame_id.to_string(),
            live,
        })
    }
}

/// Whether an enabled output fires on its interval or after a forced epoch reset.
/// An interval of zero always disables the output.
fn emit_due(frame_count: u32, every: u32, force: bool) -> bool {
    every != 0 && (force || frame_count.is_multiple_of(every))
}

/// Odometry samples kept for cloud-stamp matching.
const POSE_BUFFER_LEN: usize = 256;

/// Max stamp gap between a cloud and the pose used to register it (s).
const POSE_MATCH_TOLERANCE_S: f64 = 0.1;

fn time_secs(t: &Time) -> f64 {
    t.sec as f64 + t.nsec as f64 * 1e-9
}

/// Append a pose sample, resetting on a changed frame or clear clock rewind and
/// evicting the oldest sample to keep the buffer bounded. Returns whether the
/// previous pose epoch was cleared.
fn push_pose(poses: &mut VecDeque<PoseSample>, sample: PoseSample) -> bool {
    let reset = poses.back().is_some_and(|last| {
        last.pose.parent_frame != sample.pose.parent_frame
            || last.pose.child_frame != sample.pose.child_frame
            || last.stamp - sample.stamp > POSE_MATCH_TOLERANCE_S
    });
    if reset {
        poses.clear();
    }
    poses.push_back(sample);
    if poses.len() > POSE_BUFFER_LEN {
        poses.pop_front();
    }
    reset
}

/// The buffered pose with the stamp nearest the cloud stamp, within tolerance.
fn nearest_pose(poses: &VecDeque<PoseSample>, stamp: f64) -> Option<MatchedPose> {
    let mut best_gap = f64::INFINITY;
    let mut best = None;
    for sample in poses {
        let gap = (sample.stamp - stamp).abs();
        if gap < best_gap {
            best_gap = gap;
            best = Some(sample);
        }
    }
    if best_gap <= POSE_MATCH_TOLERANCE_S {
        best.map(|sample| sample.pose.clone())
    } else {
        None
    }
}

enum CloudPoseMatch {
    /// No exact pose and no later-pose watermark yet, so keep the cloud pending.
    Pending,
    Matched(MatchedPose),
    /// A later pose was observed, but no pose is within the fallback tolerance.
    Miss,
}

/// Prefer an exact stamp. The bounded-nearest fallback is only final once a
/// later pose proves the consumer has crossed the cloud timestamp.
fn match_pose_for_cloud(poses: &VecDeque<PoseSample>, stamp: f64) -> CloudPoseMatch {
    if let Some(sample) = poses.iter().find(|sample| sample.stamp == stamp) {
        return CloudPoseMatch::Matched(sample.pose.clone());
    }
    if !poses.iter().any(|sample| sample.stamp > stamp) {
        return CloudPoseMatch::Pending;
    }
    nearest_pose(poses, stamp).map_or(CloudPoseMatch::Miss, CloudPoseMatch::Matched)
}

struct ExtractError(&'static str);
impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

fn extract_xyz(msg: &PointCloud2) -> Result<Vec<(f32, f32, f32)>, ExtractError> {
    let mut x_off: Option<usize> = None;
    let mut y_off: Option<usize> = None;
    let mut z_off: Option<usize> = None;
    for f in &msg.fields {
        if f.datatype != PointField::FLOAT32 as u8 {
            continue;
        }
        match f.name.as_str() {
            "x" => x_off = Some(f.offset as usize),
            "y" => y_off = Some(f.offset as usize),
            "z" => z_off = Some(f.offset as usize),
            _ => {}
        }
    }
    let xo = x_off.ok_or(ExtractError("missing float32 x field"))?;
    let yo = y_off.ok_or(ExtractError("missing float32 y field"))?;
    let zo = z_off.ok_or(ExtractError("missing float32 z field"))?;

    let n = (msg.width as usize) * (msg.height as usize);
    let step = msg.point_step as usize;
    if step == 0 {
        return Err(ExtractError("point_step is 0"));
    }
    if msg.data.len() < n * step {
        return Err(ExtractError(
            "data buffer shorter than width*height*point_step",
        ));
    }
    if xo + 4 > step || yo + 4 > step || zo + 4 > step {
        return Err(ExtractError(
            "xyz field offsets do not fit within point_step",
        ));
    }
    if msg.is_bigendian {
        return Err(ExtractError("big-endian point data not supported"));
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * step;
        let x = read_f32_le(&msg.data, base + xo);
        let y = read_f32_le(&msg.data, base + yo);
        let z = read_f32_le(&msg.data, base + zo);
        if x.is_finite() && y.is_finite() && z.is_finite() {
            out.push((x, y, z));
        }
    }
    Ok(out)
}

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    let bytes: [u8; 4] = buf[off..off + 4]
        .try_into()
        .expect("bounds checked by caller");
    f32::from_le_bytes(bytes)
}

fn write_point(data: &mut Vec<u8>, n: &mut i32, x: f32, y: f32, z: f32) {
    data.extend_from_slice(&x.to_le_bytes());
    data.extend_from_slice(&y.to_le_bytes());
    data.extend_from_slice(&z.to_le_bytes());
    data.extend_from_slice(&0.0_f32.to_le_bytes());
    *n += 1;
}

fn make_cloud(data: Vec<u8>, n: i32, frame_id: &str, stamp: Time) -> PointCloud2 {
    let make_field = |name: &str, off: i32| PointField {
        name: name.into(),
        offset: off,
        datatype: PointField::FLOAT32 as u8,
        count: 1,
    };
    PointCloud2 {
        header: Header {
            seq: 0,
            stamp,
            frame_id: frame_id.into(),
        },
        height: 1,
        width: n,
        fields: vec![
            make_field("x", 0),
            make_field("y", 4),
            make_field("z", 8),
            make_field("intensity", 12),
        ],
        is_bigendian: false,
        point_step: 16,
        row_step: 16 * n,
        data,
        is_dense: true,
    }
}

/// Pack selected points into an LCM cloud message.
fn points_to_cloud(points: &[(f32, f32, f32)], frame_id: &str, stamp: Time) -> PointCloud2 {
    let mut data = Vec::with_capacity(points.len() * 16);
    let mut n: i32 = 0;
    for &(x, y, z) in points {
        write_point(&mut data, &mut n, x, y, z);
    }
    make_cloud(data, n, frame_id, stamp)
}

async fn publish_cloud(out: &Output<PointCloud2>, cloud: &PointCloud2) {
    if let Err(e) = out.publish(cloud).await {
        error_throttled!(
            Duration::from_secs(1),
            error = %e,
            topic = %out.topic,
            "Voxel map failed to publish",
        );
    }
}

#[tokio::main]
async fn main() {
    run_with_transport::<RayTracingVoxelMap>().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashSet;
    use dimos_voxel_ray_tracing::voxel_ray_tracer::Voxel;

    fn pose_sample(stamp: f64, x: f32) -> PoseSample {
        PoseSample {
            stamp,
            pose: MatchedPose {
                translation: Vector3::new(x, 0.0, 0.0),
                rotation: UnitQuaternion::identity(),
                parent_frame: "odom".to_string(),
                child_frame: "lidar".to_string(),
            },
        }
    }

    #[tokio::test]
    async fn latest_slot_handles_pre_notify_and_replaces_while_consumer_is_busy() {
        let slot = Arc::new(LatestSlot::default());
        slot.replace(1);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), slot.next())
                .await
                .expect("pre-notified value should be ready"),
            1
        );

        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let worker_slot = Arc::clone(&slot);
        let worker = tokio::spawn(async move {
            let first = worker_slot.next().await;
            started_tx
                .send(())
                .expect("test receiver should remain open");
            release_rx.await.expect("test sender should remain open");
            (first, worker_slot.next().await)
        });

        slot.replace(2);
        tokio::time::timeout(Duration::from_secs(1), started_rx)
            .await
            .expect("consumer should receive the first value")
            .expect("consumer should signal that it is busy");
        slot.replace(3);
        slot.replace(4);
        release_tx
            .send(())
            .expect("busy consumer should remain alive");

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), worker)
                .await
                .expect("consumer should finish")
                .expect("consumer task should not panic"),
            (2, 4)
        );
    }

    #[test]
    fn nearest_pose_picks_by_stamp_and_gates_on_tolerance() {
        let poses = VecDeque::from([
            pose_sample(1.0, 1.0),
            pose_sample(2.0, 2.0),
            pose_sample(3.0, 3.0),
        ]);
        let pose = nearest_pose(&poses, 2.04).expect("within tolerance");
        assert_eq!(
            pose.translation.x, 2.0,
            "nearest stamp wins, not the latest"
        );
        assert_eq!(pose.parent_frame, "odom");
        assert!(
            nearest_pose(&poses, 3.5).is_none(),
            "stale poses must not register a cloud"
        );
        assert!(nearest_pose(&VecDeque::new(), 1.0).is_none());
    }

    #[test]
    fn cloud_pose_match_waits_for_watermark_then_prefers_exact() {
        let mut poses = VecDeque::from([pose_sample(1.96, 1.0)]);
        assert!(matches!(
            match_pose_for_cloud(&poses, 2.0),
            CloudPoseMatch::Pending
        ));

        poses.push_back(pose_sample(2.08, 3.0));
        let CloudPoseMatch::Matched(fallback) = match_pose_for_cloud(&poses, 2.0) else {
            panic!("later pose should enable bounded-nearest fallback");
        };
        assert_eq!(fallback.translation.x, 1.0);

        poses.push_back(pose_sample(2.0, 2.0));
        let CloudPoseMatch::Matched(exact) = match_pose_for_cloud(&poses, 2.0) else {
            panic!("exact pose should win even after the watermark");
        };
        assert_eq!(exact.translation.x, 2.0);

        let miss = VecDeque::from([pose_sample(1.8, 1.0), pose_sample(2.2, 2.0)]);
        assert!(matches!(
            match_pose_for_cloud(&miss, 2.0),
            CloudPoseMatch::Miss
        ));
    }

    #[test]
    fn push_pose_evicts_oldest_beyond_capacity() {
        let mut poses = VecDeque::new();
        for i in 0..(POSE_BUFFER_LEN + 10) {
            assert!(!push_pose(&mut poses, pose_sample(i as f64, 0.0)));
        }
        assert_eq!(
            poses.len(),
            POSE_BUFFER_LEN,
            "buffer capped at POSE_BUFFER_LEN"
        );
        assert_eq!(poses.front().unwrap().stamp, 10.0, "oldest 10 evicted");
        assert_eq!(poses.back().unwrap().stamp, (POSE_BUFFER_LEN + 9) as f64);
    }

    #[test]
    fn map_epoch_rejects_small_reordering_and_resets_on_discontinuity() {
        let mut state = MapState::default();
        assert!(matches!(
            state.prepare_for_cloud("odom", 10.0),
            Some(CloudEpoch::Current)
        ));
        state.map.voxels.insert((0, 0, 0), Voxel::with_health(1));

        assert!(state.prepare_for_cloud("odom", 10.0).is_none());
        assert!(state.prepare_for_cloud("odom", 9.95).is_none());
        assert_eq!(state.map.voxels.len(), 1);

        assert!(matches!(
            state.prepare_for_cloud("odom", 9.0),
            Some(CloudEpoch::New)
        ));
        assert!(state.map.voxels.is_empty());
        state.map.voxels.insert((0, 0, 0), Voxel::with_health(1));

        assert!(matches!(
            state.prepare_for_cloud("map", 9.01),
            Some(CloudEpoch::New)
        ));
        assert!(state.map.voxels.is_empty());
        assert_eq!(state.frame_id.as_deref(), Some("map"));
    }

    fn cloud_points(c: &PointCloud2) -> AHashSet<(u32, u32, u32)> {
        let mut out = AHashSet::new();
        let step = c.point_step as usize;
        for i in 0..c.width as usize {
            let base = i * step;
            let x = f32::from_le_bytes(c.data[base..base + 4].try_into().unwrap());
            let y = f32::from_le_bytes(c.data[base + 4..base + 8].try_into().unwrap());
            let z = f32::from_le_bytes(c.data[base + 8..base + 12].try_into().unwrap());
            out.insert((x.to_bits(), y.to_bits(), z.to_bits()));
        }
        out
    }

    fn voxel_center(kx: i32, ky: i32, kz: i32) -> (u32, u32, u32) {
        (
            (kx as f32 + 0.5).to_bits(),
            (ky as f32 + 0.5).to_bits(),
            (kz as f32 + 0.5).to_bits(),
        )
    }

    #[test]
    fn emit_due_fires_on_interval_or_force_and_zero_disables() {
        assert!(emit_due(1, 1, false));
        assert!(emit_due(2, 1, false));
        assert!(!emit_due(1, 2, false));
        assert!(emit_due(2, 2, false));
        assert!(!emit_due(5, 3, false));
        assert!(emit_due(6, 3, false));
        assert!(emit_due(1, 5, true));
        for n in 1..10 {
            assert!(!emit_due(n, 0, false));
            assert!(!emit_due(n, 0, true));
        }
    }

    #[test]
    fn local_map_includes_voxel_inside_cylinder() {
        let mut map = VoxelMap::default();
        map.voxels.insert((0, 0, 0), Voxel::with_health(1));
        let live: AHashSet<VoxelKey> = AHashSet::new();
        let cylinder = LocalBounds {
            origin_x: 0.0,
            origin_y: 0.0,
            r_xy_max_sq: 4.0,
            z_min: 0.0,
            z_max: 1.0,
        };
        let global = points_to_cloud(
            &emit_points(&map, 1.0, None, 0, &live),
            "world",
            Time::default(),
        );
        let local = points_to_cloud(
            &emit_points(&map, 1.0, Some(&cylinder), 0, &live),
            "world",
            Time::default(),
        );
        assert!(cloud_points(&global).contains(&voxel_center(0, 0, 0)));
        assert!(cloud_points(&local).contains(&voxel_center(0, 0, 0)));
    }

    #[test]
    fn local_map_excludes_voxel_outside_radius() {
        let mut map = VoxelMap::default();
        map.voxels.insert((5, 0, 0), Voxel::with_health(1));
        let live: AHashSet<VoxelKey> = AHashSet::new();
        let cylinder = LocalBounds {
            origin_x: 0.0,
            origin_y: 0.0,
            r_xy_max_sq: 4.0,
            z_min: -10.0,
            z_max: 10.0,
        };
        let global = points_to_cloud(
            &emit_points(&map, 1.0, None, 0, &live),
            "world",
            Time::default(),
        );
        let local = points_to_cloud(
            &emit_points(&map, 1.0, Some(&cylinder), 0, &live),
            "world",
            Time::default(),
        );
        assert!(cloud_points(&global).contains(&voxel_center(5, 0, 0)));
        assert!(!cloud_points(&local).contains(&voxel_center(5, 0, 0)));
        assert_eq!(local.width, 0);
    }

    #[test]
    fn local_map_excludes_voxel_outside_z_range() {
        let mut map = VoxelMap::default();
        map.voxels.insert((0, 0, 5), Voxel::with_health(1));
        let live: AHashSet<VoxelKey> = AHashSet::new();
        let cylinder = LocalBounds {
            origin_x: 0.0,
            origin_y: 0.0,
            r_xy_max_sq: 100.0,
            z_min: 0.0,
            z_max: 1.0,
        };
        let global = points_to_cloud(
            &emit_points(&map, 1.0, None, 0, &live),
            "world",
            Time::default(),
        );
        let local = points_to_cloud(
            &emit_points(&map, 1.0, Some(&cylinder), 0, &live),
            "world",
            Time::default(),
        );
        assert!(cloud_points(&global).contains(&voxel_center(0, 0, 5)));
        assert!(!cloud_points(&local).contains(&voxel_center(0, 0, 5)));
        assert_eq!(local.width, 0);
    }

    #[test]
    fn live_voxels_follow_the_cylinder_in_local_map() {
        let map = VoxelMap::default();
        let mut live: AHashSet<VoxelKey> = AHashSet::new();
        live.insert((1, 0, 0));
        live.insert((10, 10, 10));
        let cylinder = LocalBounds {
            origin_x: 0.0,
            origin_y: 0.0,
            r_xy_max_sq: 4.0,
            z_min: 0.0,
            z_max: 1.0,
        };
        let global = points_to_cloud(
            &emit_points(&map, 1.0, None, 0, &live),
            "world",
            Time::default(),
        );
        let local = points_to_cloud(
            &emit_points(&map, 1.0, Some(&cylinder), 0, &live),
            "world",
            Time::default(),
        );
        assert!(cloud_points(&global).contains(&voxel_center(1, 0, 0)));
        assert!(cloud_points(&global).contains(&voxel_center(10, 10, 10)));
        assert!(cloud_points(&local).contains(&voxel_center(1, 0, 0)));
        assert!(!cloud_points(&local).contains(&voxel_center(10, 10, 10)));
    }

    #[test]
    fn local_map_applies_support_min() {
        // The live local cloud must honor support_min, so an isolated healthy
        // voxel is dropped while a dense patch survives. Live voxels bypass it.
        let mut map = VoxelMap::default();
        for x in 0..3 {
            for y in 0..3 {
                map.voxels.insert((x, y, 0), Voxel::with_health(1));
            }
        }
        map.voxels.insert((20, 0, 0), Voxel::with_health(1));
        let mut live: AHashSet<VoxelKey> = AHashSet::new();
        live.insert((25, 0, 0));
        let cylinder = LocalBounds {
            origin_x: 0.0,
            origin_y: 0.0,
            r_xy_max_sq: 1e6,
            z_min: -10.0,
            z_max: 10.0,
        };
        let local = points_to_cloud(
            &emit_points(&map, 1.0, Some(&cylinder), 3, &live),
            "world",
            Time::default(),
        );
        let pts = cloud_points(&local);
        assert!(pts.contains(&voxel_center(1, 1, 0)), "dense patch kept");
        assert!(
            !pts.contains(&voxel_center(20, 0, 0)),
            "isolated healthy voxel dropped by support_min"
        );
        assert!(
            pts.contains(&voxel_center(25, 0, 0)),
            "live voxel bypasses support_min"
        );
    }
}
