# Bundled Cartesian control mathematics

`core.h` is the Eigen-only, C++14-compatible shared control/spline core. It is
bundled here so the robot controller builds independently of the planner machine.
It has no Drake or ROS dependency.

Source: `drakecpp/include/cartesian/core.h`, copied on 2026-10-16.
Snapshot SHA-256: `cd10308af0a55c78cd4c8af1670442c618b5b04c988875e4ba42dbabca2da717`.
The two copies must remain identical when the planner/controller equations change.
The planner's standalone core tests are also bundled in `tests/cartesian_core_test.cc`.

On a development machine with both checkouts, sync using:

```bash
cp ../drakecpp/include/cartesian/core.h include/cartesian/core.h
cp ../drakecpp/tests/cartesian_core_test.cc tests/cartesian_core_test.cc
```

Update this checksum when syncing. The robot workstation only needs this
repository plus its existing ROS/libfranka/Eigen toolchain.
