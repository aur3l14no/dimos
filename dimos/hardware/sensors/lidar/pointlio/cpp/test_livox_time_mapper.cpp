// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

#include "livox_time_mapper.hpp"

#include <cmath>
#include <cstdio>

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool check_near(double actual, double expected, const char* message) {
    return check(std::abs(actual - expected) < 1e-9, message);
}

bool check_mapped_time(
    const std::optional<double>& actual,
    double expected,
    const char* message
) {
    if (!check(actual.has_value(), message)) { return false; }
    return check_near(*actual, expected, message);
}

}  // namespace

int main() {
    using pointlio::LivoxTimeMapper;
    using pointlio::LivoxTimeResult;

    bool ok = true;
    LivoxTimeMapper boot_time;
    ok &= check(
        boot_time.observe(0, 10'000'000'000, 1'000'000'000'000) ==
            LivoxTimeResult::kAccepted,
        "boot-relative time must establish one host anchor"
    );
    ok &= check_mapped_time(
        boot_time.to_system_seconds(10.25),
        1'000.25,
        "boot-relative cadence must map through the fixed anchor"
    );
    ok &= check(
        boot_time.observe(0, 10'250'000'000, 2'000'000'000'000) ==
            LivoxTimeResult::kAccepted,
        "later boot-relative packets must be accepted"
    );
    ok &= check_mapped_time(
        boot_time.to_system_seconds(10.5),
        1'000.5,
        "later host wall time must not retime the sensor stream"
    );
    ok &= check(
        boot_time.observe(0, 9'500'000'000, 2'001'000'000'000) ==
            LivoxTimeResult::kAccepted,
        "minor callback reordering must stay in the same epoch"
    );
    ok &= check(
        boot_time.observe(0, 2'000'000'000, 2'002'000'000'000) ==
            LivoxTimeResult::kClockRewound,
        "a device-clock rewind must be rejected"
    );
    ok &= check(
        boot_time.observe(1, 10'500'000'000, 2'003'000'000'000) ==
            LivoxTimeResult::kDomainChanged,
        "switching from boot time to PTP must require a restart"
    );

    LivoxTimeMapper ptp_time;
    ok &= check(
        ptp_time.observe(1, 1'800'000'000'000'000'000, 10) ==
            LivoxTimeResult::kAccepted,
        "PTP time must be accepted"
    );
    ok &= check_mapped_time(
        ptp_time.to_system_seconds(1'800'000'000.25),
        1'800'000'000.25,
        "PTP time must remain absolute"
    );

    LivoxTimeMapper gps_time;
    ok &= check(
        gps_time.observe(2, 1'700'000'000'000'000'000, 10) ==
            LivoxTimeResult::kAccepted,
        "GPS time must be accepted"
    );
    ok &= check_mapped_time(
        gps_time.to_system_seconds(1'700'000'000.5),
        1'700'000'000.5,
        "GPS time must remain absolute"
    );

    LivoxTimeMapper unsupported;
    ok &= check(
        unsupported.observe(3, 1, 1) == LivoxTimeResult::kUnsupportedDomain,
        "unknown Livox time domains must fail closed"
    );

    return ok ? 0 : 1;
}
