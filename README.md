# franka_effort_controllers

Joint torque controllers for the Franka Emika Panda (ros_control + `franka_hw`).

This package now has an **online trajectory-tracking mode**: instead of building the
reference trajectory locally from CSV files, the controller subscribes to a solved
trajectory published by an external planner (e.g. the MPC solver) and tracks it in
real time. New trajectories can be published back-to-back without stopping or
restarting the controller.

---

## Online trajectory tracking (`TorquePDController_Simpson`)

Controller type: `panda_mpc/TorquePDController_Simpson`

### How it works

1. A planner publishes a `trajectory_msgs/JointTrajectory` on `/reference_trajectory`.
2. `trajectoryCallback()` (non-real-time thread) validates it, maps the incoming
   joint order onto the controller's canonical order **by name** (order-independent),
   fits the splines, and writes them into a `RealtimeBuffer`:
   - `q`, `v` → cubic Hermite splines (accelerations derived from central
     differences of velocity at the knots),
   - `effort` → linear spline (the feedforward torque `tau_ff`).
3. `update()` (1 kHz real-time loop) picks up the newest trajectory by pointer
   comparison and tracks it with PD + feedforward:

   ```
   tau_i = tau_ff_i + Kp_i * (q_d_i - q_i) + Kd_i * (v_d_i - dq_filtered_i)
   ```

   - `dq_filtered_` is a low-pass filtered measured velocity (coefficient `alpha`).
   - The torque is clamped to joint torque limits and rate-limited
     (`SaturateTorqueRate`, `delta_tau_max_ = 1.0` Nm/cycle).

### Trajectory lifecycle / behavior

- **Hot swap:** publishing a new trajectory while one is being tracked switches to
  it on the next control cycle (`t_traj_` resets to 0). No controller restart.
- **Handoff safety gate:** a new trajectory is only accepted if it connects to the
  robot's current state:
  - `|q_now - q0| <= q_start_tolerance` (first knot must match current position),
  - `|v0| <= v_start_tolerance` (trajectory must start at rest).
  Otherwise the controller logs a warning and **stays put** (keeps holding the
  previous reference) until a consistent trajectory arrives.
- **End-of-trajectory hold:** once `t_traj_ >= duration`, the reference is
  deliberately frozen at the final knot (`q_d` = final pose, `v_d` = 0, `tau_ff` =
  final effort) — the robot parks at the final pose until the next trajectory is
  published. A one-time `INFO` log announces entry into the hold phase.
- **No trajectory yet:** if the controller is started before anything has been
  published, `update()` holds **zero commanded torque** (the FCI adds its own
  gravity compensation, so the arm stays put) and warns at 1 Hz until one arrives.

### Trajectory message contract

The planner must publish `trajectory_msgs/JointTrajectory` satisfying:

| Requirement | Why |
|---|---|
| All 7 joint names present: `panda_joint1` … `panda_joint7` | callback maps by name |
| `time_from_start` strictly increasing, ≥ 2 points | spline fitting / finite differences |
| `positions`, `velocities` for every point | Hermite interpolation |
| `effort` = feedforward torque **excluding gravity** | the FCI compensates gravity internally; including `g(q)` would double-compensate |
| First point ≈ robot's current `q` with `v = 0` | handoff safety gate / no torque kick |
| Both ends have zero joint velocity | clean stop and hold between segments |
| Final `effort` ≈ 0 | correct static hold at the final pose |

### ROS interface

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/reference_trajectory` | `trajectory_msgs/JointTrajectory` | sub | solved trajectory to track |
| `/torque_comparison` | `std_msgs/Float64MultiArray` | pub | commanded torque, for debugging |
| `/controller_t_start` | `std_msgs/Float64` (latched) | pub | wall-clock time of controller start |
| `/trajectory_completion` | `std_msgs/Bool` (latched) | pub | `true` once, `duration + 0.1 s` after a trajectory starts; `false` on start/stop |

The external loop can wait for `/trajectory_completion == true` before solving and
publishing the next segment.

### Parameters

Loaded from the YAML file under `panda_torque_pd_controller_simpson`
(`config/panda_mpc.yaml`). Defaults shown in parentheses.

| Parameter | Default | Meaning |
|---|---|---|
| `kp_gains` | `[40, …]` | PD position gains (7 values) |
| `kd_gains` | `[40, …]` | PD velocity gains (7 values) |
| `alpha` | `0.99` | low-pass coefficient of the measured joint velocity |
| `N` | `20` | number of knots (informational: console banner only in this controller) |
| `message_to_console` | `"Tracking with Simpson's, N = …"` | banner printed on start |
| `use_t_varying_gains` | `false` | compute `Kp = wn²·M_diag`, `Kd = 2ζ·wn·M_diag` from the mass matrix each cycle |
| `zeta` | `0.7` | damping ratio (only with `use_t_varying_gains`) |
| `natural_frequency` | `2π·4` | bandwidth (only with `use_t_varying_gains`) |
| `q_start_tolerance` | `0.05` rad | max allowed `|q_now − q0|` at handoff |
| `v_start_tolerance` | `0.1` rad/s | max allowed `|v0|` at handoff |

### Running

```bash
catkin_make            # package name: panda_torque_controllers
roslaunch panda_torque_controllers panda_launch.launch
```

The launch file starts `franka_control` and loads (without starting) the controller.
Start it with:

```bash
rosservice call /controller_manager/switch_controller \
  "start_controllers: ['panda_torque_pd_controller_simpson']
   stop_controllers: []
   strictness: 2"
```

Then publish solved trajectories on `/reference_trajectory`; the controller picks
each one up automatically and signals completion on `/trajectory_completion`.

---

## Build note

`trajectory_msgs` is used by the online version and must be declared in
`package.xml` (it is: `<depend>trajectory_msgs</depend>`). If `catkin_package`
complains about a `find_package`-ed dependency missing from `package.xml`, that is
the pattern to follow.

## Cartesian trajectory controller

The optional `panda_mpc/PandaCartesianTrajectoryController` tracks translation of the offset dummy1 tip and supports integrated joint resets. Legacy remains available. The Eigen-only `include/cartesian/core.h` is bundled in this package, so building on the robot workstation requires neither Drake nor a sibling `drakecpp` checkout. Select `panda_cartesian_trajectory_controller` explicitly at rest. The loaded `ee_offset` and any configured `q_nominal` must agree with the planner. Restart after parameter changes or a latched controller fault.

See the [Cartesian operating guide](../drakecpp/tools/README_cartesian.md) for messages, controller-start acknowledgements, telemetry, precision validation and workstation build requirements. The current implementation has offline mathematical/ROS message tests; its plugin build and actual robot tracking require workstation validation.

## Simpson controller recordings (joint/Cartesian references and torques)

Recording is enabled in `config/panda_mpc.yaml`. Rebuild this package on the robot
workstation, then reload the controller (or restart its launch) before collecting
strikes. No changes to the planner/live-loop repository are required.

```bash
# Run in the ROS catkin workspace on the robot workstation.
catkin_make --pkg panda_torque_controllers
```

At controller load, the console prints `Simpson recording directory: ...`.
The default is `~/.ros/franka_effort_controllers/simpson_<UTC timestamp>_<pid>/`
on **the machine running franka_control**, not necessarily the planner laptop.
Use `recording_directory` to choose another writable local directory. Each load
creates a unique session; controller restarts within that load share it.

The files are:

- `settings.json`: loaded gain/filter settings, configured static gains, torque
  limits, the current per-cycle torque-rate limit, handoff tolerances, tip offset,
  interpolation description, and recording settings. `alpha` weights the newest
  measured velocity; 0.99 provides very little filtering.
- `reference_<trajectory_id>.csv`: received knots in canonical Panda joint order:
  `t,q1..7,v1..7,u1..7,a_interpolation1..7`. The acceleration columns are the actual
  finite differences used to construct the independent velocity spline. These
  knots reproduce the controller's current reference exactly.
- `reference_<trajectory_id>.json`: original message sequence/stamp/frame and
  receipt timestamp. Receiving a reference does not imply it was executed.
- `executions.csv`: controller-start and per-execution trajectory-start ROS
  timestamps, first sample timestamp, duration, trajectory ID and execution ID.
  Hot-swapped trajectories and replays receive distinct execution IDs. IDs are
  local to this controller load and are not `strike_iterNNN` numbers; compare
  the reference knots/source stamp to associate them with planner CSVs.
- `samples.csv`: one row per controller cycle while executing, plus the configured
  `recording_hold_seconds` (default 2 s) of final hold. Its header names every
  column. It includes timestamps, period, reference time, measured `q`, raw and
  filtered `dq`, evaluated `q_ref`/`dq_ref`, the diagnostic `dq_from_q_ref`, actual
  per-cycle `kp`/`kd`, feedforward/position/velocity torque components, each torque
  stage, and limiter bit masks (bit 0 = joint 1). The gains recorded here include
  the configuration-dependent mass-diagonal scaling.
- `status.json`: written on normal controller unload/shutdown after draining the
  buffer; reports written/dropped samples and write success. Its absence means
  orderly shutdown was not confirmed. While running, inspect the sequence gaps,
  `dropped_samples`, and console warnings. Samples/executions flush every 0.5 s;
  abrupt process termination can lose the latest unflushed data.

Torque columns have deliberately distinct meanings:

| Columns | Meaning |
|---|---|
| `tau_requested1..7` | Feedforward + PD before either controller limiter |
| `tau_clamped1..7` | After absolute joint-torque clipping |
| `tau_command1..7` | After rate limiting; value sent through the effort interface |
| `tau_J_d1..7` | Franka-reported desired joint torque, excluding gravity; observed before this cycle's new command |
| `tau_J1..7` | Measured joint torque sensor signals, including gravity effects |

The effort-interface command is not a direct measurement of motor-applied torque.
FCI/hardware processing can change it; compare it with subsequent `tau_J_d`
samples, allowing for transport/control delay. Do not directly subtract the
measured `tau_J` from a gravity-excluded command to infer tracking error.

With `recording_cartesian: true`, each sample also contains `p_ref_x/y/z`,
`v_ref_x/y/z`, `p_measured_x/y/z`, and `v_measured_x/y/z`. These are computed from
Franka's model in the robot base frame, for a point at `recording_tip_offset`
relative to the flange (`panda_link8`). Set that offset to match the planner's
`ee_offset`; the initial value is `[0, 0, 0.3384]` metres. The measured tip state is
FK/Jacobian-derived from measured joints, not an external position measurement.
The velocity reference uses the controller's actual independent velocity spline.
The Franka model may differ slightly from the Drake model used by Meshcat.
Setting `recording_cartesian: false` writes NaNs in the Cartesian columns and
avoids the extra FK/Jacobian evaluations.

All CSV formatting and disk I/O run outside `update()` on a background thread.
The control thread copies into a fixed-capacity buffer; overflow drops samples
and increments a counter rather than blocking control. Recording attempts every
control cycle (normally 1 kHz), not the existing slow `/torque_comparison` topic.
Cartesian logging adds model evaluations to the control cycle; inspect `period`
and workstation timing when validating this instrumentation. The logger does
not change gain values, filtering, interpolation, terminal feedforward, command
admission, or torque limiting. Invalid trajectory arrays are rejected before
interpolation/logging rather than throwing on indexing.

The existing live loop already saves `strike_iterNNN.csv` with planned joint
states/torques and reconstructs the displayed Cartesian path from those knots;
it does not save a separate dense Cartesian trace. These controller recordings
add the evaluated Cartesian reference and measured motion on the same clock as
the actual command. For analysis, copy the whole controller session directory
alongside the corresponding planner run directory.

The standalone recorder test can run without a robot or ROS master:

```bash
g++ -std=c++14 -pthread -I include tests/simpson_recording_test.cc -o /tmp/simpson_recording_test
/tmp/simpson_recording_test
```


## Building on a separate robot workstation

The controller runs on the PC directly connected to the robot. The planner and
Meshcat can run on another computer; the two communicate through ROS. Install or
sync **this entire controller repository**, including `include/cartesian/core.h`,
into the remote catkin workspace (`src/franka_effort_controllers-online` is fine).
The ROS package name is `panda_torque_controllers`, independent of that folder name.
No Drake installation or planner checkout is needed to compile these controllers.

After syncing the changes, run these commands on the robot PC (SSH or its remote
terminal):

```bash
source /opt/ros/noetic/setup.bash
cd /home/sc19zx/catkin_ws
catkin_make --pkg panda_torque_controllers
source devel/setup.bash
```

Reload the controller/restart its launch at rest to load the rebuilt library.
There is no need to set `CARTESIAN_CORE_INCLUDE_DIR`; stale values in an existing
CMake cache are ignored. The empty `generate_messages()` call and its explicit
zero-message target dependencies have been removed; this package consumes ROS
messages and does not generate custom messages.

Controller recordings are saved on the robot PC. With the default configuration
and user `sc19zx`, they are under `/home/sc19zx/.ros/franka_effort_controllers/`.
Use `recording_directory` to change that location. Copy the printed session
folder back to the analysis machine together with the planner's run directory.

The bundled Cartesian core is a byte-identical snapshot of the planner's shared
header. When its equations change, sync the header to both repositories and run
the shared-core tests; see `include/cartesian/README.md` for provenance.
