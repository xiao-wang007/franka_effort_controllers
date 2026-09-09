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
roslaunch franka_effort_controllers panda_launch.launch
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
