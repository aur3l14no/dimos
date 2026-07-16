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

use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::PointCloud2;
use lcm_msgs::std_msgs::Time;
use nalgebra::{UnitQuaternion, Vector3};

/// Odometry samples kept for cloud-stamp matching.
const POSE_BUFFER_LEN: usize = 256;

/// Max stamp gap between a cloud and the pose used to register it.
const POSE_MATCH_TOLERANCE_NS: i64 = 100_000_000;

#[derive(Clone)]
struct PoseSample {
    stamp: Time,
    translation: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
    parent_frame: String,
    child_frame: String,
}

#[derive(Debug, PartialEq)]
pub(crate) enum FrameContractError {
    EmptyParentFrame,
    CloudFrameMismatch,
}

#[derive(Debug, PartialEq)]
pub(crate) enum PoseDropReason {
    MissingExactPose,
    OutsideTolerance,
    Frame(FrameContractError),
}

enum PoseMatch {
    Pending,
    Drop(PoseDropReason),
    Matched(PoseSample),
}

pub(crate) struct MatchedCloudPose {
    pub(crate) cloud: PointCloud2,
    pub(crate) translation: Vector3<f32>,
    pub(crate) rotation: UnitQuaternion<f32>,
    pub(crate) parent_frame: String,
}

pub(crate) enum MatchOutcome {
    Pending,
    Drop(PoseDropReason),
    Matched(MatchedCloudPose),
}

pub(crate) struct CloudMatch {
    pub(crate) replaced_pending: bool,
    pub(crate) outcome: MatchOutcome,
}

/// Owns the pose history and cloud that must advance together during matching.
#[derive(Default)]
pub(crate) struct CloudPoseMatcher {
    poses: VecDeque<PoseSample>,
    pending_cloud: Option<PointCloud2>,
}

impl CloudPoseMatcher {
    pub(crate) fn push_odometry(
        &mut self,
        msg: Odometry,
        require_exact_pose_match: bool,
    ) -> MatchOutcome {
        let p = &msg.pose.pose.position;
        let q = &msg.pose.pose.orientation;
        let sample = PoseSample {
            stamp: msg.header.stamp,
            translation: Vector3::new(p.x as f32, p.y as f32, p.z as f32),
            rotation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                q.w as f32, q.x as f32, q.y as f32, q.z as f32,
            )),
            parent_frame: msg.header.frame_id,
            child_frame: msg.child_frame_id,
        };
        if push_pose(&mut self.poses, sample) {
            self.pending_cloud = None;
        }
        self.try_match(require_exact_pose_match)
    }

    pub(crate) fn push_cloud(
        &mut self,
        cloud: PointCloud2,
        require_exact_pose_match: bool,
    ) -> CloudMatch {
        let replaced_pending = self.pending_cloud.replace(cloud).is_some();
        CloudMatch {
            replaced_pending,
            outcome: self.try_match(require_exact_pose_match),
        }
    }

    fn try_match(&mut self, require_exact_pose_match: bool) -> MatchOutcome {
        let Some(cloud) = self.pending_cloud.as_ref() else {
            return MatchOutcome::Pending;
        };
        let pose = match match_pose(&self.poses, cloud, require_exact_pose_match) {
            PoseMatch::Pending => return MatchOutcome::Pending,
            PoseMatch::Drop(reason) => {
                self.pending_cloud = None;
                return MatchOutcome::Drop(reason);
            }
            PoseMatch::Matched(pose) => pose,
        };
        let cloud = self
            .pending_cloud
            .take()
            .expect("pending cloud was matched above");
        MatchOutcome::Matched(MatchedCloudPose {
            cloud,
            translation: pose.translation,
            rotation: pose.rotation,
            parent_frame: pose.parent_frame,
        })
    }
}

fn time_nanos(t: &Time) -> i64 {
    i64::from(t.sec) * 1_000_000_000 + i64::from(t.nsec)
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

/// Match only after an exact sample arrives or a newer pose establishes a
/// watermark. This keeps a cloud pending while an exact sample may still arrive.
fn match_pose(
    poses: &VecDeque<PoseSample>,
    cloud: &PointCloud2,
    require_exact_pose_match: bool,
) -> PoseMatch {
    if let Some(pose) = poses
        .iter()
        .rev()
        .find(|pose| pose.stamp == cloud.header.stamp)
    {
        return validate_matched_pose(pose, cloud);
    }

    let cloud_stamp_ns = time_nanos(&cloud.header.stamp);
    if !poses
        .iter()
        .any(|pose| time_nanos(&pose.stamp) > cloud_stamp_ns)
    {
        return PoseMatch::Pending;
    }
    if require_exact_pose_match {
        return PoseMatch::Drop(PoseDropReason::MissingExactPose);
    }

    let mut best_gap = u64::MAX;
    let mut best = None;
    for pose in poses {
        let gap = (time_nanos(&pose.stamp) - cloud_stamp_ns).unsigned_abs();
        if gap < best_gap {
            best_gap = gap;
            best = Some(pose);
        }
    }
    if best_gap <= POSE_MATCH_TOLERANCE_NS as u64 {
        validate_matched_pose(best.expect("a newer pose established the watermark"), cloud)
    } else {
        PoseMatch::Drop(PoseDropReason::OutsideTolerance)
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

fn validate_matched_pose(pose: &PoseSample, cloud: &PointCloud2) -> PoseMatch {
    match validate_frames(pose, cloud) {
        Ok(()) => PoseMatch::Matched(pose.clone()),
        Err(error) => PoseMatch::Drop(PoseDropReason::Frame(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcm_msgs::std_msgs::Header;

    fn stamp(milliseconds: i32) -> Time {
        Time {
            sec: milliseconds / 1_000,
            nsec: (milliseconds % 1_000) * 1_000_000,
        }
    }

    fn pose(milliseconds: i32, x: f32, parent_frame: &str, child_frame: &str) -> PoseSample {
        PoseSample {
            stamp: stamp(milliseconds),
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
    fn pose_match_waits_for_watermark_and_prefers_exact_stamp() {
        let mut poses = VecDeque::new();
        push_pose(&mut poses, pose(1_950, 1.0, "odom", "lidar"));
        let pointcloud = cloud(2_000, "lidar");

        assert!(matches!(
            match_pose(&poses, &pointcloud, false),
            PoseMatch::Pending
        ));

        push_pose(&mut poses, pose(2_000, 2.0, "odom", "lidar"));
        push_pose(&mut poses, pose(2_080, 3.0, "odom", "lidar"));
        let PoseMatch::Matched(exact) = match_pose(&poses, &pointcloud, false) else {
            panic!("exact pose should match")
        };
        assert_eq!(exact.translation.x, 2.0);

        let later_cloud = cloud(2_050, "lidar");
        let PoseMatch::Matched(nearest) = match_pose(&poses, &later_cloud, false) else {
            panic!("watermark should allow a nearest-pose match")
        };
        assert_eq!(nearest.translation.x, 3.0);

        let far_cloud = cloud(3_000, "lidar");
        assert!(matches!(
            match_pose(&poses, &far_cloud, false),
            PoseMatch::Pending
        ));
        push_pose(&mut poses, pose(3_200, 4.0, "odom", "lidar"));
        assert!(matches!(
            match_pose(&poses, &far_cloud, false),
            PoseMatch::Drop(PoseDropReason::OutsideTolerance)
        ));
    }

    #[test]
    fn exact_only_pose_match_drops_after_watermark() {
        let exact_cloud = cloud(1_000, "lidar");
        let exact_poses = VecDeque::from([pose(1_000, 0.0, "odom", "lidar")]);
        assert!(matches!(
            match_pose(&exact_poses, &exact_cloud, true),
            PoseMatch::Matched(_)
        ));

        let mut poses = VecDeque::new();
        let pointcloud = cloud(2_000, "lidar");
        push_pose(&mut poses, pose(1_950, 1.0, "odom", "lidar"));
        assert!(matches!(
            match_pose(&poses, &pointcloud, true),
            PoseMatch::Pending
        ));

        push_pose(&mut poses, pose(2_080, 2.0, "odom", "lidar"));
        assert!(matches!(
            match_pose(&poses, &pointcloud, true),
            PoseMatch::Drop(PoseDropReason::MissingExactPose)
        ));
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
        assert_eq!(poses.front().unwrap().stamp, stamp(10), "oldest 10 evicted");
        assert_eq!(
            poses.back().unwrap().stamp,
            stamp((POSE_BUFFER_LEN + 9) as i32)
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
}
