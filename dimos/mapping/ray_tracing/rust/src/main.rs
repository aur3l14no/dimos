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
    batch_local_bounds, emit_points, update_map, Config, LocalBounds, VoxelMap,
};
use lcm_msgs::geometry_msgs::{Point, Pose, PoseStamped, Quaternion};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{PointCloud2, PointField};
use lcm_msgs::std_msgs::{Header, Time};
use nalgebra::{UnitQuaternion, Vector3};
use tokio::sync::Notify;

struct MapJob {
    cloud: PointCloud2,
    translation: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
    parent_frame: String,
}

#[derive(Clone)]
struct PoseSample {
    stamp: f64,
    translation: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
    parent_frame: String,
    child_frame: String,
}

#[derive(Debug, PartialEq)]
enum FrameContractError {
    EmptyParentFrame,
    CloudFrameMismatch,
}

struct LatestState<T> {
    pending: Option<T>,
    replaced: u64,
}

impl<T> Default for LatestState<T> {
    fn default() -> Self {
        Self {
            pending: None,
            replaced: 0,
        }
    }
}

/// A single-consumer slot that retains only the newest pending value.
struct LatestSlot<T> {
    state: Mutex<LatestState<T>>,
    wake: Notify,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(LatestState::default()),
            wake: Notify::new(),
        }
    }
}

impl<T> LatestSlot<T> {
    /// Replace the pending value, returning the cumulative replacement count
    /// when an older value was displaced.
    fn replace(&self, value: T) -> Option<u64> {
        let replaced = {
            let mut state = self.state.lock().expect("latest slot mutex");
            if state.pending.replace(value).is_some() {
                state.replaced += 1;
                Some(state.replaced)
            } else {
                None
            }
        };
        self.wake.notify_one();
        replaced
    }

    fn take(&self) -> Option<T> {
        self.state.lock().expect("latest slot mutex").pending.take()
    }

    async fn next(&self) -> T {
        loop {
            if let Some(value) = self.take() {
                return value;
            }
            self.wake.notified().await;
        }
    }

    #[cfg(test)]
    fn replaced_count(&self) -> u64 {
        self.state.lock().expect("latest slot mutex").replaced
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
        let sample = PoseSample {
            stamp: time_secs(&msg.header.stamp),
            translation: Vector3::new(p.x as f32, p.y as f32, p.z as f32),
            rotation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                q.w as f32, q.x as f32, q.y as f32, q.z as f32,
            )),
            parent_frame: msg.header.frame_id,
            child_frame: msg.child_frame_id,
        };
        push_pose(&mut self.poses, sample);
    }

    async fn on_lidar(&mut self, msg: PointCloud2) {
        self.ensure_worker_running().await;
        let Some(pose) = nearest_pose(&self.poses, time_secs(&msg.header.stamp)) else {
            warn_throttled!(
                Duration::from_secs(1),
                "No odometry within tolerance of the cloud stamp, dropped a cloud.",
            );
            return;
        };
        if let Err(error) = validate_frames(&pose, &msg) {
            warn_throttled!(
                Duration::from_secs(1),
                error = ?error,
                "Cloud and odometry frames are incompatible; dropped a cloud.",
            );
            return;
        }
        if let Some(replaced_total) = self.latest_job.replace(MapJob {
            cloud: msg,
            translation: pose.translation,
            rotation: pose.rotation,
            parent_frame: pose.parent_frame,
        }) {
            warn_throttled!(
                Duration::from_secs(1),
                replaced_total,
                "Ray tracing is busy; replaced the pending cloud with a newer one.",
            );
        }
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
    parent_frame: Option<String>,
}

impl MapState {
    /// Reset before accepting points expressed in a different parent frame.
    /// Returns whether the map was reset.
    fn prepare_for_parent(&mut self, parent_frame: &str) -> bool {
        let parent_changed = self
            .parent_frame
            .as_deref()
            .is_some_and(|current| current != parent_frame);
        if parent_changed {
            *self = Self {
                parent_frame: Some(parent_frame.to_string()),
                ..Self::default()
            };
            return true;
        }

        self.parent_frame
            .get_or_insert_with(|| parent_frame.to_string());
        false
    }
}

struct MapOutputs {
    bounds: Option<PoseStamped>,
    global: Option<PointCloud2>,
    local: Option<PointCloud2>,
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
        let Some(outputs) = tokio::task::block_in_place(|| self.update(state, job)) else {
            return;
        };

        if let Some(bounds) = outputs.bounds {
            if let Err(error) = self.region_bounds.publish(&bounds).await {
                error_throttled!(
                    Duration::from_secs(1),
                    error = %error,
                    "Region bounds failed to publish",
                );
            }
        }
        if let Some(global) = outputs.global {
            publish_cloud(&self.global_map, &global).await;
        }
        if let Some(local) = outputs.local {
            publish_cloud(&self.local_map, &local).await;
        }
    }

    fn update(&self, state: &mut MapState, job: MapJob) -> Option<MapOutputs> {
        let MapJob {
            cloud,
            translation,
            rotation,
            parent_frame,
        } = job;
        let origin = (translation.x, translation.y, translation.z);
        let voxel_size = self.config.voxel_size;

        let points = match extract_xyz(&cloud) {
            Ok(points) => points,
            Err(error) => {
                warn_throttled!(
                    Duration::from_secs(1),
                    error = %error,
                    "Failed to get lidar points, dropped a cloud.",
                );
                return None;
            }
        };
        if points.is_empty() {
            return None;
        }
        let parent_changed = state.prepare_for_parent(&parent_frame);

        // Transform sensor-frame points into the odometry parent frame.
        let rot = rotation.to_rotation_matrix();
        let points: Vec<(f32, f32, f32)> = points
            .iter()
            .map(|&(x, y, z)| {
                let point = rot * Vector3::new(x, y, z) + translation;
                (point.x, point.y, point.z)
            })
            .collect();

        let out_frame_id = parent_frame.as_str();
        let live = update_map(&mut state.map, origin, &points, &self.config);

        // The batch only feeds the local region bounds, so skip it when the local
        // map is disabled.
        if self.config.emit_every > 0 {
            state.batch_points.extend_from_slice(&points);
            state.batch_origins.push(origin);
        }

        state.frame_count += 1;
        let local_due =
            emit_due_or_frame_change(state.frame_count, self.config.emit_every, parent_changed);
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

        let stamp = cloud.header.stamp;
        let global_due = emit_due_or_frame_change(
            state.frame_count,
            self.config.global_emit_every,
            parent_changed,
        );
        let global = global_due.then(|| {
            let points = emit_points(&state.map, voxel_size, None, 0, &live);
            points_to_cloud(&points, out_frame_id, stamp.clone())
        });
        let local = cylinder.as_ref().map(|bounds| {
            let points = emit_points(
                &state.map,
                voxel_size,
                Some(bounds),
                self.config.support_min,
                &live,
            );
            points_to_cloud(&points, out_frame_id, stamp)
        });

        Some(MapOutputs {
            bounds,
            global,
            local,
        })
    }
}

/// Whether the Nth-frame output fires this frame. Zero disables it.
fn emit_due(frame_count: u32, every: u32) -> bool {
    every != 0 && frame_count.is_multiple_of(every)
}

/// A new coordinate epoch must replace the previously published map without
/// waiting for the next periodic emission. Zero still disables the output.
fn emit_due_or_frame_change(frame_count: u32, every: u32, frame_changed: bool) -> bool {
    every != 0 && (frame_changed || emit_due(frame_count, every))
}

/// Odometry samples kept for cloud-stamp matching.
const POSE_BUFFER_LEN: usize = 256;

/// Max stamp gap between a cloud and the pose used to register it (s).
const POSE_MATCH_TOLERANCE_S: f64 = 0.1;

fn time_secs(t: &Time) -> f64 {
    t.sec as f64 + t.nsec as f64 * 1e-9
}

/// Append a pose sample, resetting on a frame epoch change.
/// Returns whether the history was reset.
fn push_pose(poses: &mut VecDeque<PoseSample>, sample: PoseSample) -> bool {
    let frames_changed = poses.front().is_some_and(|current| {
        current.parent_frame != sample.parent_frame || current.child_frame != sample.child_frame
    });
    if frames_changed {
        poses.clear();
    }

    poses.push_back(sample);
    if poses.len() > POSE_BUFFER_LEN {
        poses.pop_front();
    }
    frames_changed
}

/// The buffered pose with the stamp nearest the cloud stamp, within tolerance.
fn nearest_pose(poses: &VecDeque<PoseSample>, stamp: f64) -> Option<PoseSample> {
    let mut best_gap = f64::INFINITY;
    let mut best = None;
    for pose in poses {
        let gap = (pose.stamp - stamp).abs();
        if gap < best_gap {
            best_gap = gap;
            best = Some(pose.clone());
        }
    }
    if best_gap <= POSE_MATCH_TOLERANCE_S {
        best
    } else {
        None
    }
}

fn validate_frames(pose: &PoseSample, cloud: &PointCloud2) -> Result<(), FrameContractError> {
    if pose.parent_frame.is_empty() {
        return Err(FrameContractError::EmptyParentFrame);
    }
    if !cloud.header.frame_id.is_empty() && cloud.header.frame_id != pose.child_frame {
        return Err(FrameContractError::CloudFrameMismatch);
    }
    Ok(())
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
    use dimos_voxel_ray_tracing::voxel_ray_tracer::{Voxel, VoxelKey};

    #[test]
    fn latest_slot_keeps_newest_value_and_counts_replacements() {
        let slot = LatestSlot::default();

        assert_eq!(slot.replace(1), None);
        assert_eq!(slot.replace(2), Some(1));
        assert_eq!(slot.replace(3), Some(2));
        assert_eq!(slot.replaced_count(), 2);
        assert_eq!(slot.take(), Some(3));

        assert_eq!(slot.replace(4), None);
        assert_eq!(slot.replaced_count(), 2);
        assert_eq!(slot.take(), Some(4));
    }

    fn stamp(milliseconds: i32) -> Time {
        Time {
            sec: milliseconds / 1_000,
            nsec: (milliseconds % 1_000) * 1_000_000,
        }
    }

    fn pose(milliseconds: i32, x: f32, parent_frame: &str, child_frame: &str) -> PoseSample {
        PoseSample {
            stamp: milliseconds as f64 / 1_000.0,
            translation: Vector3::new(x, 0.0, 0.0),
            rotation: UnitQuaternion::identity(),
            parent_frame: parent_frame.to_string(),
            child_frame: child_frame.to_string(),
        }
    }

    fn cloud(milliseconds: i32, frame_id: &str) -> PointCloud2 {
        PointCloud2 {
            header: Header {
                stamp: stamp(milliseconds),
                frame_id: frame_id.to_string(),
                ..Header::default()
            },
            ..PointCloud2::default()
        }
    }

    #[test]
    fn nearest_pose_picks_by_stamp_and_gates_on_tolerance() {
        let mut poses = VecDeque::new();
        push_pose(&mut poses, pose(1_950, 1.0, "odom", "lidar"));
        push_pose(&mut poses, pose(2_000, 2.0, "odom", "lidar"));
        push_pose(&mut poses, pose(2_080, 3.0, "odom", "lidar"));

        let exact = nearest_pose(&poses, 2.0).expect("exact pose should match");
        assert_eq!(exact.translation.x, 2.0);
        let nearest = nearest_pose(&poses, 2.05).expect("nearest pose should match");
        assert_eq!(nearest.translation.x, 3.0);
        assert!(nearest_pose(&poses, 3.0).is_none());
        assert!(nearest_pose(&VecDeque::new(), 1.0).is_none());
    }

    #[test]
    fn frame_contract_enforces_parent_and_cloud_frames() {
        let pointcloud = cloud(1_000, "lidar");
        let mut sample = pose(1_000, 0.0, "", "lidar");
        assert_eq!(
            validate_frames(&sample, &pointcloud),
            Err(FrameContractError::EmptyParentFrame)
        );

        sample = pose(1_000, 0.0, "odom", "base");
        assert_eq!(
            validate_frames(&sample, &pointcloud),
            Err(FrameContractError::CloudFrameMismatch)
        );
        assert_eq!(validate_frames(&sample, &cloud(1_000, "")), Ok(()));
    }

    #[test]
    fn push_pose_evicts_oldest_beyond_capacity() {
        let mut poses = VecDeque::new();
        for i in 0..(POSE_BUFFER_LEN + 10) {
            push_pose(&mut poses, pose(i as i32, 0.0, "odom", "lidar"));
        }
        assert_eq!(
            poses.len(),
            POSE_BUFFER_LEN,
            "buffer capped at POSE_BUFFER_LEN"
        );
        assert_eq!(poses.front().unwrap().stamp, 0.01, "oldest 10 evicted");
        assert_eq!(
            poses.back().unwrap().stamp,
            (POSE_BUFFER_LEN + 9) as f64 / 1_000.0
        );
    }

    #[test]
    fn pose_history_resets_on_frame_epoch() {
        let mut poses = VecDeque::new();
        assert!(!push_pose(&mut poses, pose(1_000, 0.0, "odom", "lidar")));
        assert!(!push_pose(&mut poses, pose(1_050, 0.0, "odom", "lidar")));

        assert!(push_pose(&mut poses, pose(1_060, 0.0, "map", "lidar")));
        assert_eq!(poses.len(), 1);
        assert_eq!(poses.front().unwrap().parent_frame, "map");

        assert!(push_pose(&mut poses, pose(1_070, 0.0, "map", "base")));
        assert_eq!(poses.len(), 1);
        assert_eq!(poses.front().unwrap().child_frame, "base");
    }

    #[test]
    fn map_state_resets_only_on_parent_frame_change() {
        let mut state = MapState::default();
        assert!(!state.prepare_for_parent("odom"));
        state.map.voxels.insert((0, 0, 0), Voxel::with_health(1));
        state.frame_count = 7;
        state.batch_points.push((1.0, 2.0, 3.0));

        assert!(!state.prepare_for_parent("odom"));
        assert!(state.map.voxels.contains_key(&(0, 0, 0)));
        assert_eq!(state.frame_count, 7);

        assert!(state.prepare_for_parent("map"));
        assert!(state.map.voxels.is_empty());
        assert_eq!(state.frame_count, 0);
        assert!(state.batch_points.is_empty());
        assert_eq!(state.parent_frame.as_deref(), Some("map"));
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
    fn emit_due_fires_every_nth_frame_and_zero_disables() {
        assert!(emit_due(1, 1));
        assert!(emit_due(2, 1));
        assert!(!emit_due(1, 2));
        assert!(emit_due(2, 2));
        assert!(!emit_due(5, 3));
        assert!(emit_due(6, 3));
        for n in 1..10 {
            assert!(!emit_due(n, 0));
        }

        assert!(emit_due_or_frame_change(1, 50, true));
        assert!(!emit_due_or_frame_change(1, 50, false));
        assert!(!emit_due_or_frame_change(1, 0, true));
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
