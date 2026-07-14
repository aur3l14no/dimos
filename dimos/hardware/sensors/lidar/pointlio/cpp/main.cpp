// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Point-LIO + Livox Mid-360 native module for dimos NativeModule framework.
//
// Binds Livox SDK2 directly into the Point-LIO core: SDK callbacks feed
// CustomMsg/Imu to the IESKF estimator, which performs LiDAR-inertial SLAM.
// Sensor-frame point clouds and odometry are published on LCM.
//
// Usage:
//   ./pointlio_native \
//       --lidar '/lidar#sensor_msgs.PointCloud2' \
//       --odometry '/odometry#nav_msgs.Odometry' \
//       --filter_size_surf 0.2 --ivox_grid_resolution 2.0 ... \   # tuning as plain CLI args
//       --host_ip 192.168.1.5 --lidar_ip 192.168.1.155

#include <lcm/lcm-cpp.hpp>

#include <atomic>
#include <boost/make_shared.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "livox_sdk_config.hpp"
#include "livox_time_mapper.hpp"

#include "dimos_native_module.hpp"

#include "geometry_msgs/Quaternion.hpp"
#include "geometry_msgs/Vector3.hpp"
#include "nav_msgs/Odometry.hpp"
#include "sensor_msgs/Imu.hpp"
#include "sensor_msgs/PointCloud2.hpp"
#include "sensor_msgs/PointField.hpp"

// Point-LIO (header-only core, compiled sources linked via CMake)
#include "pointlio.hpp"
#include "pointlio_debug.hpp"

using livox_common::GRAVITY_MS2;
using livox_common::DATA_TYPE_IMU;
using livox_common::DATA_TYPE_CARTESIAN_HIGH;
using livox_common::DATA_TYPE_CARTESIAN_LOW;

static std::atomic<bool> g_running{true};
static lcm::LCM* g_lcm = nullptr;
static PointLio* g_point_lio = nullptr;

static uint64_t system_time_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

// Parse a comma-separated list of doubles (CLI vector args); empty on bad input.
static std::vector<double> parse_doubles(const std::string& csv) {
    std::vector<double> out;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) { j = csv.size(); }
        try {
            out.push_back(std::stod(csv.substr(i, j - i)));
        } catch (...) {
            return {};
        }
        i = j + 1;
    }
    return out;
}

static std::string g_lidar_topic;
static std::string g_deskewed_lidar_topic;
static std::string g_odometry_topic;
static std::string g_lidar_odometry_topic;
static std::string g_frame_id;          // required via --frame_id
static std::string g_sensor_frame_id;    // required via --sensor_frame_id
static float g_frequency = 10.0f;

// Frame accumulator (Livox SDK raw → CustomMsg)
static std::mutex g_pc_mutex;
// Serializes all Point-LIO EKF access. The SDK delivers IMU on its own callback
// thread (on_imu_data → feed_imu) while the main loop runs feed_lidar/process/
// get_* — Point-LIO's estimator is not thread-safe, so without this the two
// threads race on the EKF state and occasionally emit a corrupt 2nd trajectory.
// Distinct from g_pc_mutex (which only guards the point accumulator) so incoming
// point packets can still accumulate while the EKF is processing.
static std::mutex g_lio_mutex;
static std::mutex g_sensor_time_mutex;
static pointlio::LivoxTimeMapper g_sensor_time_mapper;
static std::vector<custom_messages::CustomPoint> g_accumulated_points;
static uint64_t g_frame_start_ns = 0;
static bool g_frame_has_timestamp = false;

static uint64_t get_timestamp_ns(const LivoxLidarEthernetPacket* pkt) {
    uint64_t ns = 0;
    std::memcpy(&ns, pkt->timestamp, sizeof(uint64_t));
    return ns;
}

static bool observe_sensor_time(const LivoxLidarEthernetPacket* packet) {
    const uint64_t sensor_ns = get_timestamp_ns(packet);
    std::lock_guard<std::mutex> lock(g_sensor_time_mutex);
    if (!g_running.load()) { return false; }

    const auto result = g_sensor_time_mapper.observe(
        packet->time_type,
        sensor_ns,
        system_time_ns()
    );
    if (result == pointlio::LivoxTimeResult::kAccepted) { return true; }

    const char* reason = result == pointlio::LivoxTimeResult::kUnsupportedDomain
        ? "unsupported time_type"
        : result == pointlio::LivoxTimeResult::kDomainChanged
            ? "time domain changed"
            : "sensor clock rewound";
    fprintf(stderr, "[pointlio] %s; restart before reusing estimator state\n", reason);
    g_running.store(false);
    return false;
}

static std::optional<double> to_system_time(double sensor_seconds) {
    std::lock_guard<std::mutex> lock(g_sensor_time_mutex);
    return g_sensor_time_mapper.to_system_seconds(sensor_seconds);
}

using dimos::make_header;

// Preserve the legacy lidar output as Point-LIO's IMU/body-frame cloud.
// deskewed_lidar instead contains filter_size_surf-downsampled, point-time-
// registered endpoints re-expressed at the final LiDAR pose; it is not a
// world-frame registered_scan.
static void publish_lidar(
    PointCloudXYZI::Ptr cloud,
    double timestamp,
    const std::string& topic
) {
    if (!g_lcm || !cloud || cloud->empty() || topic.empty()) { return; }

    int num_points = static_cast<int>(cloud->size());

    sensor_msgs::PointCloud2 pc;
    pc.header = make_header(g_sensor_frame_id, timestamp);
    pc.height = 1;
    pc.width = num_points;
    pc.is_bigendian = 0;
    pc.is_dense = 1;

    pc.fields_length = 4;
    pc.fields.resize(4);

    auto make_field = [](const std::string& name, int32_t offset) {
        sensor_msgs::PointField field;
        field.name = name;
        field.offset = offset;
        field.datatype = sensor_msgs::PointField::FLOAT32;
        field.count = 1;
        return field;
    };

    pc.fields[0] = make_field("x", 0);
    pc.fields[1] = make_field("y", 4);
    pc.fields[2] = make_field("z", 8);
    pc.fields[3] = make_field("intensity", 12);

    pc.point_step = 16;
    pc.row_step = pc.point_step * num_points;

    pc.data_length = pc.row_step;
    pc.data.resize(pc.data_length);

    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        float* dst = reinterpret_cast<float*>(pc.data.data() + point_idx * 16);
        dst[0] = cloud->points[point_idx].x;
        dst[1] = cloud->points[point_idx].y;
        dst[2] = cloud->points[point_idx].z;
        dst[3] = cloud->points[point_idx].intensity;
    }

    g_lcm->publish(topic, &pc);
}

static void publish_odometry(
    const custom_messages::Odometry& odom,
    double timestamp,
    const std::string& topic
) {
    if (!g_lcm || topic.empty()) { return; }

    nav_msgs::Odometry msg;
    msg.header = make_header(g_frame_id, timestamp);
    msg.child_frame_id = g_sensor_frame_id;

    // The legacy stream preserves Point-LIO's core pose; lidar_odometry carries
    // the dedicated final LiDAR pose for deskewed_lidar.
    msg.pose.pose.position.x = odom.pose.pose.position.x;
    msg.pose.pose.position.y = odom.pose.pose.position.y;
    msg.pose.pose.position.z = odom.pose.pose.position.z;
    msg.pose.pose.orientation.x = odom.pose.pose.orientation.x;
    msg.pose.pose.orientation.y = odom.pose.pose.orientation.y;
    msg.pose.pose.orientation.z = odom.pose.pose.orientation.z;
    msg.pose.pose.orientation.w = odom.pose.pose.orientation.w;

    for (int idx = 0; idx < 36; ++idx) {
        msg.pose.covariance[idx] = odom.pose.covariance[idx];
    }

    // Preserve Point-LIO's estimator twist convention. This is not adjusted
    // for the LiDAR/IMU lever arm when the pose child frame is the LiDAR.
    msg.twist.twist.linear.x = odom.twist.twist.linear.x;
    msg.twist.twist.linear.y = odom.twist.twist.linear.y;
    msg.twist.twist.linear.z = odom.twist.twist.linear.z;
    msg.twist.twist.angular.x = odom.twist.twist.angular.x;
    msg.twist.twist.angular.y = odom.twist.twist.angular.y;
    msg.twist.twist.angular.z = odom.twist.twist.angular.z;
    std::memset(msg.twist.covariance, 0, sizeof(msg.twist.covariance));

    g_lcm->publish(topic, &msg);
}


static void on_point_cloud(const uint32_t /*handle*/, const uint8_t /*dev_type*/, LivoxLidarEthernetPacket* data, void* /*client_data*/) {
    if (!g_running.load() || data == nullptr) { return; }
    if (!observe_sensor_time(data)) { return; }

    uint64_t ts_ns = get_timestamp_ns(data);
    uint16_t dot_num = data->dot_num;

    // Per-point intra-packet offset (matches livox_ros_driver2). Without it all
    // points share one timestamp and per-point deskew is lost. time_interval
    // unit is 0.1us, so *100 → ns.
    const uint64_t point_interval_ns = dot_num > 0 ? static_cast<uint64_t>(data->time_interval) * 100 / dot_num : 0;

    std::lock_guard<std::mutex> lock(g_pc_mutex);

    if (!g_frame_has_timestamp) {
        g_frame_start_ns = ts_ns;
        g_frame_has_timestamp = true;
    }
    if (ts_ns < g_frame_start_ns) { return; }

    if (data->data_type == DATA_TYPE_CARTESIAN_HIGH) {
        auto* pts = reinterpret_cast<const LivoxLidarCartesianHighRawPoint*>(data->data);
        for (uint16_t point_idx = 0; point_idx < dot_num; ++point_idx) {
            custom_messages::CustomPoint cp;
            cp.x = static_cast<double>(pts[point_idx].x) / 1000.0;   // mm → m
            cp.y = static_cast<double>(pts[point_idx].y) / 1000.0;
            cp.z = static_cast<double>(pts[point_idx].z) / 1000.0;
            cp.reflectivity = pts[point_idx].reflectivity;
            cp.tag = pts[point_idx].tag;
            cp.line = 0;  // Mid-360: single line
            cp.offset_time = static_cast<uli>((ts_ns - g_frame_start_ns) + point_idx * point_interval_ns);
            g_accumulated_points.push_back(cp);
        }
    } else if (data->data_type == DATA_TYPE_CARTESIAN_LOW) {
        auto* pts = reinterpret_cast<const LivoxLidarCartesianLowRawPoint*>(data->data);
        for (uint16_t point_idx = 0; point_idx < dot_num; ++point_idx) {
            custom_messages::CustomPoint cp;
            cp.x = static_cast<double>(pts[point_idx].x) / 100.0;   // cm → m
            cp.y = static_cast<double>(pts[point_idx].y) / 100.0;
            cp.z = static_cast<double>(pts[point_idx].z) / 100.0;
            cp.reflectivity = pts[point_idx].reflectivity;
            cp.tag = pts[point_idx].tag;
            cp.line = 0;
            cp.offset_time = static_cast<uli>((ts_ns - g_frame_start_ns) + point_idx * point_interval_ns);
            g_accumulated_points.push_back(cp);
        }
    }
}

static void on_imu_data(const uint32_t /*handle*/, const uint8_t /*dev_type*/, LivoxLidarEthernetPacket* data, void* /*client_data*/) {
    if (!g_running.load() || data == nullptr || !g_point_lio) { return; }
    if (!observe_sensor_time(data)) { return; }

    uint64_t pkt_ts_ns = get_timestamp_ns(data);
    double ts = static_cast<double>(pkt_ts_ns) / 1e9;
    auto* imu_pts = reinterpret_cast<const LivoxLidarImuRawPoint*>(data->data);
    uint16_t dot_num = data->dot_num;

    // Serialize EKF access against the main loop (run_main_iter). Held across the
    // whole packet so its samples feed atomically.
    std::lock_guard<std::mutex> lio_lock(g_lio_mutex);
    for (uint16_t point_idx = 0; point_idx < dot_num; ++point_idx) {
        auto imu_msg = boost::make_shared<custom_messages::Imu>();
        imu_msg->header.stamp = custom_messages::Time().fromSec(ts);
        imu_msg->header.seq = 0;
        imu_msg->header.frame_id = "livox_frame";

        imu_msg->orientation.x = 0.0;
        imu_msg->orientation.y = 0.0;
        imu_msg->orientation.z = 0.0;
        imu_msg->orientation.w = 1.0;
        for (int cov_idx = 0; cov_idx < 9; ++cov_idx) {
            imu_msg->orientation_covariance[cov_idx] = 0.0;
        }

        imu_msg->angular_velocity.x = static_cast<double>(imu_pts[point_idx].gyro_x);
        imu_msg->angular_velocity.y = static_cast<double>(imu_pts[point_idx].gyro_y);
        imu_msg->angular_velocity.z = static_cast<double>(imu_pts[point_idx].gyro_z);
        for (int cov_idx = 0; cov_idx < 9; ++cov_idx) {
            imu_msg->angular_velocity_covariance[cov_idx] = 0.0;
        }

        // Point-LIO expects accel in g (EKF does its own scaling). SDK already
        // reports g, so feed raw — scaling by GRAVITY_MS2 would double-scale and
        // trip the satu_acc check at rest.
        imu_msg->linear_acceleration.x = static_cast<double>(imu_pts[point_idx].acc_x);
        imu_msg->linear_acceleration.y = static_cast<double>(imu_pts[point_idx].acc_y);
        imu_msg->linear_acceleration.z = static_cast<double>(imu_pts[point_idx].acc_z);
        for (int cov_idx = 0; cov_idx < 9; ++cov_idx) {
            imu_msg->linear_acceleration_covariance[cov_idx] = 0.0;
        }

        g_point_lio->feed_imu(imu_msg);
    }
}

static void on_info_change(const uint32_t handle, const LivoxLidarInfo* info, void* /*client_data*/) {
    if (info == nullptr) { return; }

    char sn[17] = {};
    std::memcpy(sn, info->sn, 16);
    char ip[17] = {};
    std::memcpy(ip, info->lidar_ip, 16);

    if (pointlio_debug) {
        printf("[pointlio] Device connected: handle=%u type=%u sn=%s ip=%s\n", handle, info->dev_type, sn, ip);
    }

    SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, nullptr, nullptr);
    EnableLivoxLidarImuData(handle, nullptr, nullptr);
}

static void signal_handler(int /*sig*/) {
    g_running.store(false);
}

int main(int argc, char** argv) {
    dimos::NativeModule mod(argc, argv);

    g_lidar_topic = mod.has("lidar") ? mod.topic("lidar") : "";
    g_deskewed_lidar_topic =
        mod.has("deskewed_lidar") ? mod.topic("deskewed_lidar") : "";
    g_odometry_topic = mod.has("odometry") ? mod.topic("odometry") : "";
    g_lidar_odometry_topic =
        mod.has("lidar_odometry") ? mod.topic("lidar_odometry") : "";

    if (g_deskewed_lidar_topic.empty() != g_lidar_odometry_topic.empty()) {
        fprintf(
            stderr,
            "Error: --deskewed_lidar and --lidar_odometry must be configured together\n"
        );
        return 1;
    }
    if (g_lidar_topic.empty() && g_deskewed_lidar_topic.empty() &&
        g_odometry_topic.empty()) {
        fprintf(stderr, "Error: at least one output topic is required\n");
        return 1;
    }

    // Point-LIO tuning, passed as CLI args by the dimos module (no YAML).
    PointLioParams params;
    // common
    params.con_frame = mod.arg_bool("con_frame", params.con_frame);
    params.con_frame_num = mod.arg_int("con_frame_num", params.con_frame_num);
    params.cut_frame = mod.arg_bool("cut_frame", params.cut_frame);
    params.cut_frame_time_interval = mod.arg_float("cut_frame_time_interval", params.cut_frame_time_interval);
    params.time_lag_imu_to_lidar = mod.arg_float("time_lag_imu_to_lidar", params.time_lag_imu_to_lidar);
    // preprocess
    params.scan_line = mod.arg_int("scan_line", params.scan_line);
    params.scan_rate = mod.arg_int("scan_rate", params.scan_rate);
    params.blind = mod.arg_float("blind", params.blind);
    params.point_filter_num = mod.arg_int("point_filter_num", params.point_filter_num);
    std::string lidar_type = mod.arg("lidar_type", "avia");
    params.lidar_type = lidar_type == "velodyne" ? 2 : lidar_type == "ouster" ? 3 :
                        lidar_type == "hesai" ? 4 : lidar_type == "unilidar" ? 5 : 1;
    std::string ts_unit = mod.arg("timestamp_unit", "nanosecond");
    params.timestamp_unit = ts_unit == "second" ? 0 : ts_unit == "millisecond" ? 1 :
                            ts_unit == "microsecond" ? 2 : 3;
    // mapping
    params.use_imu_as_input = mod.arg_bool("use_imu_as_input", params.use_imu_as_input);
    params.prop_at_freq_of_imu = mod.arg_bool("prop_at_freq_of_imu", params.prop_at_freq_of_imu);
    params.check_satu = mod.arg_bool("check_satu", params.check_satu);
    params.init_map_size = mod.arg_int("init_map_size", params.init_map_size);
    params.space_down_sample = mod.arg_bool("space_down_sample", params.space_down_sample);
    params.satu_acc = mod.arg_float("satu_acc", params.satu_acc);
    params.satu_gyro = mod.arg_float("satu_gyro", params.satu_gyro);
    params.acc_norm = mod.arg_float("acc_norm", params.acc_norm);
    params.plane_thr = mod.arg_float("plane_thr", params.plane_thr);
    params.filter_size_surf = mod.arg_float("filter_size_surf", params.filter_size_surf);
    params.filter_size_map = mod.arg_float("filter_size_map", params.filter_size_map);
    params.ivox_grid_resolution = mod.arg_float("ivox_grid_resolution", params.ivox_grid_resolution);
    std::string ivox_nearby = mod.arg("ivox_nearby_type", "nearby6");
    params.ivox_nearby_type = ivox_nearby == "center" ? 0 : ivox_nearby == "nearby18" ? 18 :
                              ivox_nearby == "nearby26" ? 26 : 6;
    params.cube_side_length = mod.arg_float("cube_side_length", params.cube_side_length);
    params.det_range = mod.arg_float("det_range", params.det_range);
    params.fov_degree = mod.arg_float("fov_degree", params.fov_degree);
    params.imu_en = mod.arg_bool("imu_en", params.imu_en);
    params.start_in_aggressive_motion = mod.arg_bool("start_in_aggressive_motion", params.start_in_aggressive_motion);
    params.extrinsic_est_en = mod.arg_bool("extrinsic_est_en", params.extrinsic_est_en);
    params.imu_time_inte = mod.arg_float("imu_time_inte", params.imu_time_inte);
    params.lidar_meas_cov = mod.arg_float("lidar_meas_cov", params.lidar_meas_cov);
    params.acc_cov_input = mod.arg_float("acc_cov_input", params.acc_cov_input);
    params.vel_cov = mod.arg_float("vel_cov", params.vel_cov);
    params.gyr_cov_input = mod.arg_float("gyr_cov_input", params.gyr_cov_input);
    params.gyr_cov_output = mod.arg_float("gyr_cov_output", params.gyr_cov_output);
    params.acc_cov_output = mod.arg_float("acc_cov_output", params.acc_cov_output);
    params.b_gyr_cov = mod.arg_float("b_gyr_cov", params.b_gyr_cov);
    params.b_acc_cov = mod.arg_float("b_acc_cov", params.b_acc_cov);
    params.imu_meas_acc_cov = mod.arg_float("imu_meas_acc_cov", params.imu_meas_acc_cov);
    params.imu_meas_omg_cov = mod.arg_float("imu_meas_omg_cov", params.imu_meas_omg_cov);
    params.match_s = mod.arg_float("match_s", params.match_s);
    params.gravity_align = mod.arg_bool("gravity_align", params.gravity_align);
    if (auto g = parse_doubles(mod.arg("gravity", "")); !g.empty()) params.gravity = g;
    if (auto gi = parse_doubles(mod.arg("gravity_init", "")); !gi.empty()) params.gravity_init = gi;
    if (auto et = parse_doubles(mod.arg("extrinsic_t", "")); !et.empty()) params.extrinsic_T = et;
    if (auto er = parse_doubles(mod.arg("extrinsic_r", "")); !er.empty()) params.extrinsic_R = er;
    // odometry
    params.publish_odometry_without_downsample =
        mod.arg_bool("publish_odometry_without_downsample", params.publish_odometry_without_downsample);
    params.odom_only = mod.arg_bool("odom_only", params.odom_only);

    // Point-LIO internal processing rates
    double msr_freq = mod.arg_float("msr_freq", 50.0f);
    double main_freq = mod.arg_float("main_freq", 5000.0f);

    // Livox hardware config
    std::string host_ip = mod.arg("host_ip", "192.168.1.5");
    std::string lidar_ip = mod.arg("lidar_ip", "192.168.1.155");
    g_frequency = mod.arg_float("frequency", 10.0f);
    g_frame_id = mod.arg_required("frame_id");
    g_sensor_frame_id = mod.arg_required("sensor_frame_id");
    float pointcloud_freq = mod.arg_float("pointcloud_freq", 5.0f);
    float odom_freq = mod.arg_float("odom_freq", 50.0f);
    bool scan_publish_en = mod.arg_bool("scan_publish_en", true);
    bool deskewed_scan_publish_en = mod.arg_bool("deskewed_scan_publish_en", false);

    // Propagates to the Point-LIO core via the `pointlio_debug` global.
    bool debug = mod.arg_bool("debug", false);
    pointlio_debug = debug;

    // SDK network ports (defaults from SdkPorts struct in livox_sdk_config.hpp)
    livox_common::SdkPorts ports;
    const livox_common::SdkPorts port_defaults;
    ports.cmd_data        = mod.arg_int("cmd_data_port", port_defaults.cmd_data);
    ports.push_msg        = mod.arg_int("push_msg_port", port_defaults.push_msg);
    ports.point_data      = mod.arg_int("point_data_port", port_defaults.point_data);
    ports.imu_data        = mod.arg_int("imu_data_port", port_defaults.imu_data);
    ports.log_data        = mod.arg_int("log_data_port", port_defaults.log_data);
    ports.host_cmd_data   = mod.arg_int("host_cmd_data_port", port_defaults.host_cmd_data);
    ports.host_push_msg   = mod.arg_int("host_push_msg_port", port_defaults.host_push_msg);
    ports.host_point_data = mod.arg_int("host_point_data_port", port_defaults.host_point_data);
    ports.host_imu_data   = mod.arg_int("host_imu_data_port", port_defaults.host_imu_data);
    ports.host_log_data   = mod.arg_int("host_log_data_port", port_defaults.host_log_data);

    if (debug) {
        printf("[pointlio] Starting Point-LIO + Livox Mid-360 native module\n");
        printf("[pointlio] lidar topic: %s\n", g_lidar_topic.empty() ? "(disabled)" : g_lidar_topic.c_str());
        printf("[pointlio] deskewed lidar topic: %s\n", g_deskewed_lidar_topic.empty() ? "(disabled)" : g_deskewed_lidar_topic.c_str());
        printf("[pointlio] odometry topic: %s\n", g_odometry_topic.empty() ? "(disabled)" : g_odometry_topic.c_str());
        printf("[pointlio] lidar odometry topic: %s\n", g_lidar_odometry_topic.empty() ? "(disabled)" : g_lidar_odometry_topic.c_str());
        printf("[pointlio] tuning: filter_size_surf=%.3f ivox_res=%.3f lidar_type=%d\n", params.filter_size_surf, params.ivox_grid_resolution, params.lidar_type);
        printf("[pointlio] host_ip: %s  lidar_ip: %s  frequency: %.1f Hz\n", host_ip.c_str(), lidar_ip.c_str(), g_frequency);
        printf("[pointlio] pointcloud_freq: %.1f Hz  odom_freq: %.1f Hz\n", pointcloud_freq, odom_freq);
        printf(
            "[pointlio] publish: legacy_scan=%s deskewed_scan=%s\n",
            scan_publish_en ? "on" : "off",
            deskewed_scan_publish_en ? "on" : "off"
        );
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "Error: LCM init failed\n");
        return 1;
    }
    g_lcm = &lcm;

    if (debug) { printf("[pointlio] Initializing Point-LIO...\n"); }
    PointLio point_lio(params, msr_freq, main_freq);
    g_point_lio = &point_lio;
    if (debug) { printf("[pointlio] Point-LIO initialized.\n"); }

    // Main-loop state. Body lives in `run_main_iter`, driven by the wall-paced
    // main thread.
    auto frame_interval = std::chrono::microseconds( static_cast<int64_t>(1e6 / g_frequency));
    std::optional<std::chrono::steady_clock::time_point> last_emit;
    const double process_period_ms = 1000.0 / main_freq;

    const auto pc_interval =
        std::chrono::microseconds(static_cast<int64_t>(1e6 / pointcloud_freq));
    const auto odom_interval =
        std::chrono::microseconds(static_cast<int64_t>(1e6 / odom_freq));
    std::optional<std::chrono::steady_clock::time_point> next_pc_publish;
    std::optional<std::chrono::steady_clock::time_point> next_odom_publish;

    // Publishing at or above the scan aggregation rate means every genuine
    // estimator result. Lower rates use an anchored deadline instead of
    // `last = now`; a small tolerance absorbs scheduler jitter without turning
    // nominal 10 Hz input/output into an accidental 5 Hz stream.
    auto publication_due = [](
        std::chrono::steady_clock::time_point now,
        bool every_result,
        std::chrono::microseconds interval,
        std::optional<std::chrono::steady_clock::time_point>& next_deadline
    ) {
        if (every_result) { return true; }
        if (!next_deadline.has_value()) {
            next_deadline = now + interval;
            return true;
        }

        const auto candidate_tolerance = interval / 20;
        const auto max_tolerance = std::chrono::microseconds(5'000);
        const auto tolerance = candidate_tolerance < max_tolerance
            ? candidate_tolerance
            : max_tolerance;
        if (now + tolerance < *next_deadline) { return false; }

        do {
            *next_deadline += interval;
        } while (*next_deadline <= now);
        return true;
    };


    auto run_main_iter = [&](std::chrono::steady_clock::time_point now) {
        // Lazy-seed rate-limit bookmarks on the first iteration so they align
        // with the wall clock.
        if (!last_emit.has_value()) {
            last_emit = now;
        }
        // At frame rate: drain accumulated points into a CustomMsg and feed
        // Point-LIO. Hold g_pc_mutex across the rate-limit check AND swap so the
        // clock + accumulator are observed atomically (no packet slips between).
        std::vector<custom_messages::CustomPoint> points;
        uint64_t frame_start = 0;
        {
            std::lock_guard<std::mutex> lock(g_pc_mutex);
            if (now - *last_emit >= frame_interval) {
                if (!g_accumulated_points.empty()) {
                    points.swap(g_accumulated_points);
                    frame_start = g_frame_start_ns;
                    g_frame_has_timestamp = false;
                }
                last_emit = now;
            }
        }
        bool odometry_due = false;
        bool has_legacy_odometry = false;
        custom_messages::Odometry legacy_odometry;
        PointCloudXYZI::Ptr lidar_cloud;
        DeskewedLidarSnapshot deskewed_snapshot;
        {
            // Serialize estimator access against the SDK IMU callback. Copy the
            // coherent result under this lock, then encode and publish after it
            // is released so packet ingestion is not blocked by point packing.
            std::lock_guard<std::mutex> lio_lock(g_lio_mutex);
            if (!points.empty()) {
                const size_t num_points = points.size();
                auto lidar_msg = boost::make_shared<custom_messages::CustomMsg>();
                lidar_msg->header.seq = 0;
                lidar_msg->header.stamp = custom_messages::Time().fromSec(
                    static_cast<double>(frame_start) / 1e9
                );
                lidar_msg->header.frame_id = "livox_frame";
                lidar_msg->timebase = frame_start;
                lidar_msg->lidar_id = 0;
                for (int idx = 0; idx < 3; idx++) { lidar_msg->rsvd[idx] = 0; }
                lidar_msg->point_num = static_cast<uli>(num_points);
                lidar_msg->points = std::move(points);
                if (pointlio_debug) {
                    fprintf(stderr, "[pointlio] feed_lidar frame: %zu points\n", num_points);
                }
                point_lio.feed_lidar(lidar_msg);
            }

            if (!point_lio.process()) { return; }

            const bool legacy_cloud_enabled =
                scan_publish_en && !g_lidar_topic.empty();
            const bool deskewed_cloud_enabled =
                deskewed_scan_publish_en && !g_deskewed_lidar_topic.empty();
            const bool cloud_due = (legacy_cloud_enabled || deskewed_cloud_enabled) &&
                publication_due(
                    now,
                    pointcloud_freq >= g_frequency,
                    pc_interval,
                    next_pc_publish
                );
            // odom_freq is a requested ceiling. process() is the new-result
            // gate, so actual odometry cannot exceed the estimator result rate.
            odometry_due = !g_odometry_topic.empty() && publication_due(
                now,
                odom_freq >= g_frequency,
                odom_interval,
                next_odom_publish
            );

            if (odometry_due || (cloud_due && legacy_cloud_enabled)) {
                legacy_odometry = point_lio.get_odometry();
                has_legacy_odometry = true;
            }
            if (cloud_due) {
                if (legacy_cloud_enabled) {
                    lidar_cloud = point_lio.get_body_cloud();
                }
                if (deskewed_cloud_enabled) {
                    deskewed_snapshot = point_lio.get_deskewed_lidar_snapshot();
                }
            }
        }

        std::optional<double> legacy_publish_time;
        if (has_legacy_odometry) {
            legacy_publish_time = to_system_time(legacy_odometry.header.stamp.toSec());
            if (!legacy_publish_time.has_value()) {
                fprintf(stderr, "[pointlio] legacy estimator time is outside the active Livox epoch\n");
                g_running.store(false);
                return;
            }
        }

        std::optional<double> deskewed_publish_time;
        if (deskewed_snapshot.cloud && !deskewed_snapshot.cloud->empty()) {
            deskewed_publish_time = to_system_time(
                deskewed_snapshot.lidar_odometry.header.stamp.toSec()
            );
            if (!deskewed_publish_time.has_value()) {
                fprintf(stderr, "[pointlio] deskewed estimator time is outside the active Livox epoch\n");
                g_running.store(false);
                return;
            }
        }

        if (!g_running.load()) { return; }

        if (lidar_cloud && !lidar_cloud->empty() && legacy_publish_time.has_value()) {
            publish_lidar(lidar_cloud, *legacy_publish_time, g_lidar_topic);
            if (pointlio_debug) {
                fprintf(
                    stderr,
                    "[pointlio] publish legacy lidar: %zu points stamp=%.9f\n",
                    lidar_cloud->size(),
                    *legacy_publish_time
                );
            }
        }
        if (deskewed_publish_time.has_value()) {
            // This pose reconstructs every registered hit endpoint exactly.
            // A downstream ray tracer still approximates all ray origins with
            // this one final reference pose unless the transport grows a
            // per-time-group origin representation.
            publish_odometry(
                deskewed_snapshot.lidar_odometry,
                *deskewed_publish_time,
                g_lidar_odometry_topic
            );
            publish_lidar(
                deskewed_snapshot.cloud,
                *deskewed_publish_time,
                g_deskewed_lidar_topic
            );
            if (pointlio_debug) {
                fprintf(
                    stderr,
                    "[pointlio] publish deskewed lidar: %zu points stamp=%.9f\n",
                    deskewed_snapshot.cloud->size(),
                    *deskewed_publish_time
                );
            }
        }

        if (odometry_due && legacy_publish_time.has_value()) {
            publish_odometry(legacy_odometry, *legacy_publish_time, g_odometry_topic);
            if (pointlio_debug) {
                fprintf(
                    stderr,
                    "[pointlio] publish odom: pose=(%.3f, %.3f, %.3f)\n",
                    legacy_odometry.pose.pose.position.x,
                    legacy_odometry.pose.pose.position.y,
                    legacy_odometry.pose.pose.position.z
                );
            }
        }
    };

    // Packet source: Livox SDK callbacks from its own threads feed the
    // accumulator/EKF; the main thread below owns run_main_iter.
    if (!livox_common::init_livox_sdk(host_ip, lidar_ip, ports, debug)) {
        return 1;
    }
    SetLivoxLidarPointCloudCallBack(on_point_cloud, nullptr);
    SetLivoxLidarImuDataCallback(on_imu_data, nullptr);
    SetLivoxLidarInfoChangeCallback(on_info_change, nullptr);
    if (!LivoxLidarSdkStart()) {
        fprintf(stderr, "Error: LivoxLidarSdkStart failed\n");
        LivoxLidarSdkUninit();
        return 1;
    }
    if (debug) { printf("[pointlio] SDK started, waiting for device...\n"); }

    while (g_running.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        run_main_iter(std::chrono::steady_clock::now());

        lcm.handleTimeout(0);

        // Rate control (~main_freq, 5kHz default).
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        if (elapsed_ms < process_period_ms) {
            std::this_thread::sleep_for(std::chrono::microseconds( static_cast<int64_t>((process_period_ms - elapsed_ms) * 1000)));
        }
    }

    if (debug) { printf("[pointlio] Shutting down...\n"); }
    // Uninit (stops + joins the SDK callback threads) BEFORE clearing the
    // pointers those callbacks read, so an in-flight on_imu_data/on_point_cloud
    // can't race the assignment and dereference a null g_point_lio / g_lcm.
    LivoxLidarSdkUninit();
    g_point_lio = nullptr;
    g_lcm = nullptr;

    if (debug) { printf("[pointlio] Done.\n"); }
    return 0;
}
