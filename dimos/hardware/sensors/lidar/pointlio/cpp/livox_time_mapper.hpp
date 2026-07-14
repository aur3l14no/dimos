// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace pointlio {

enum class LivoxTimeResult {
    kAccepted,
    kUnsupportedDomain,
    kDomainChanged,
    kClockRewound,
};

// Livox time_type 0 is device-boot-relative; 1 (PTP/gPTP) and 2 (GPS)
// already carry an absolute nanosecond timestamp. A process must stay in one
// domain because changing epochs while retaining estimator state is unsafe.
// Callers serialize access to this object.
class LivoxTimeMapper {
public:
    // Point and IMU callbacks may arrive slightly out of order. This tolerance
    // admits that scheduling jitter without re-anchoring; an older timestamp
    // beyond it is treated as a new device epoch and requires a restart.
    static constexpr uint64_t kRewindToleranceNanoseconds = 1'000'000'000;

    LivoxTimeResult observe(uint8_t time_type, uint64_t sensor_ns, uint64_t system_ns) {
        if (time_type > 2) {
            return LivoxTimeResult::kUnsupportedDomain;
        }
        if (time_type_.has_value() && time_type != *time_type_) {
            return LivoxTimeResult::kDomainChanged;
        }
        if (latest_sensor_ns_.has_value() && sensor_ns < *latest_sensor_ns_ &&
            *latest_sensor_ns_ - sensor_ns > kRewindToleranceNanoseconds) {
            return LivoxTimeResult::kClockRewound;
        }

        if (!time_type_.has_value()) {
            time_type_ = time_type;
            sensor_anchor_ns_ = sensor_ns;
            system_anchor_ns_ = system_ns;
        }
        latest_sensor_ns_ = latest_sensor_ns_.has_value()
            ? std::max(*latest_sensor_ns_, sensor_ns)
            : sensor_ns;
        return LivoxTimeResult::kAccepted;
    }

    std::optional<double> to_system_seconds(double sensor_seconds) const {
        if (!time_type_.has_value() || !std::isfinite(sensor_seconds) || sensor_seconds < 0.0) {
            return std::nullopt;
        }
        if (*time_type_ != 0) {
            return sensor_seconds;
        }

        const long double anchor_sensor_seconds =
            static_cast<long double>(sensor_anchor_ns_) / 1'000'000'000.0L;
        const long double anchor_system_seconds =
            static_cast<long double>(system_anchor_ns_) / 1'000'000'000.0L;
        const long double mapped = anchor_system_seconds +
            static_cast<long double>(sensor_seconds) - anchor_sensor_seconds;
        const double result = static_cast<double>(mapped);
        return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
    }

private:
    std::optional<uint8_t> time_type_;
    std::optional<uint64_t> latest_sensor_ns_;
    uint64_t sensor_anchor_ns_ = 0;
    uint64_t system_anchor_ns_ = 0;
};

}  // namespace pointlio
