#include <cartesian/core.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Header.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <urdf/model.h>

#include <atomic>
#include <memory>

namespace franka_torque_controller {
using namespace cartesian;
class PandaCartesianTrajectoryController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaModelInterface, franka_hw::FrankaStateInterface,
          hardware_interface::EffortJointInterface> {
  struct Command {
    bool valid = false, joint = false;
    uint32_t id = 0;
    uint64_t generation = 0;
    Trajectory trajectory;
    Quintic<7> joint_trajectory;
    double duration = 0., blend = 1.;
  };
  struct Sample {
    std::array<double, 51> x;
  };
  static constexpr size_t capacity = 8192;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_;
  std::array<hardware_interface::JointHandle, 7> joints_;
  realtime_tools::RealtimeBuffer<Command> buffer_;
  Command active_;
  std::atomic<uint64_t> generation_{0}, write_{0}, read_{0}, dropped_{0};
  std::atomic<int> admission_status_{0};
  std::atomic<uint32_t> admission_id_{0};
  std::array<Sample, capacity> samples_;
  ros::Subscriber cart_sub_, joint_sub_;
  ros::Publisher telemetry_pub_;
  ros::Timer telemetry_timer_;
  realtime_tools::RealtimePublisher<std_msgs::Header> start_pub_;
  realtime_tools::RealtimePublisher<std_msgs::Bool> completion_pub_;
  Gains gains_;
  V3 offset_ = V3(0, 0, .3384), initial_axis_ = -V3::UnitZ(),
     hold_axis_ = -V3::UnitZ();
  V7 filtered_ = V7::Zero(), nominal_ = V7::Zero(),
     configured_nominal_ = V7::Zero(), hold_q_ = V7::Zero();
  V7 lower_, upper_, velocity_limit_;
  bool has_nominal_ = false, holding_ = true, fault_ = false,
       start_pending_ = false, completion_pending_ = false, completed_ = false;
  double elapsed_ = 0., started_ = 0., rest_time_ = 0.;
  double start_position_tol_ = .001, start_joint_tol_ = .05, rest_tol_ = .05;
  uint64_t seen_ = 0, cycles_ = 0;

  static V3 Translation(const geometry_msgs::Vector3& x) {
    return V3(x.x, x.y, x.z);
  }
  static bool Finite(const State<3>& x) {
    return x.p.allFinite() && x.v.allFinite() && x.a.allFinite();
  }
  static bool ZeroAngular(const geometry_msgs::Twist& x) {
    return Translation(x.angular).norm() < 1e-12;
  }
  void Cartesian(
      const trajectory_msgs::MultiDOFJointTrajectory::ConstPtr& msg) {
    const size_t count = msg->points.size();
    if ((count != 2 && count != 4) || msg->header.frame_id != "panda_link0" ||
        msg->joint_names != std::vector<std::string>{"dummy1"}) {
      ROS_WARN(
          "Cartesian command has invalid frame, tip name or boundary count");
      return;
    }
    Command cmd;
    cmd.id = msg->header.seq;
    cmd.trajectory.count = count - 1;
    std::array<State<3>, 4> x;
    std::array<double, 4> t{};
    for (size_t i = 0; i < count; ++i) {
      const auto& pt = msg->points[i];
      if (pt.transforms.size() != 1 || pt.velocities.size() != 1 ||
          pt.accelerations.size() != 1) {
        ROS_WARN(
            "Cartesian boundary needs transform, velocity and acceleration");
        return;
      }
      const auto& R = pt.transforms[0].rotation;
      if (!std::isfinite(R.x) || !std::isfinite(R.y) || !std::isfinite(R.z) ||
          !std::isfinite(R.w) ||
          std::abs(R.x) + std::abs(R.y) + std::abs(R.z) > 1e-12 ||
          std::abs(R.w - 1) > 1e-12 || !ZeroAngular(pt.velocities[0]) ||
          !ZeroAngular(pt.accelerations[0])) {
        ROS_WARN("Cartesian commands must contain translation only");
        return;
      }
      x[i].p = Translation(pt.transforms[0].translation);
      x[i].v = Translation(pt.velocities[0].linear);
      x[i].a = Translation(pt.accelerations[0].linear);
      t[i] = pt.time_from_start.toSec();
      if (!Finite(x[i]) || !std::isfinite(t[i]) ||
          (i == 0 ? t[i] != 0 : t[i] <= t[i - 1])) {
        ROS_WARN("Invalid Cartesian boundary state/time");
        return;
      }
    }
    if (x[0].v.norm() > 1e-8 || x[0].a.norm() > 1e-8 ||
        x[count - 1].v.norm() > 1e-8 || x[count - 1].a.norm() > 1e-8 ||
        t[count - 1] > 20.) {
      ROS_WARN("Cartesian trajectory must start and finish at rest");
      return;
    }
    for (size_t i = 0; i + 1 < count; ++i)
      cmd.trajectory.segments[i] =
          Quintic<3>::Fit(x[i], x[i + 1], t[i + 1] - t[i]);
    cmd.duration = t[count - 1];
    cmd.blend = count == 4 ? t[1] : cmd.duration;
    if (count == 4) {
      cmd.trajectory.contact_time = t[1];
      cmd.trajectory.carry_end_time = t[2];
    }
    cmd.valid = true;
    cmd.generation = ++generation_;
    buffer_.writeFromNonRT(cmd);
  }
  void Joint(const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
    if (msg->points.size() != 2 || msg->joint_names.size() != 7) {
      ROS_WARN(
          "Joint reset needs exactly two boundaries and seven joint names");
      return;
    }
    std::array<int, 7> order;
    for (int j = 0; j < 7; ++j) {
      const std::string name = "panda_joint" + std::to_string(j + 1);
      auto it =
          std::find(msg->joint_names.begin(), msg->joint_names.end(), name);
      if (it == msg->joint_names.end()) {
        ROS_WARN("Joint reset names incomplete");
        return;
      }
      order[j] = it - msg->joint_names.begin();
    }
    State<7> a, b;
    const auto& p0 = msg->points[0];
    const auto& p1 = msg->points[1];
    for (const auto* pt : {&p0, &p1})
      if (pt->positions.size() != 7 || pt->velocities.size() != 7 ||
          pt->accelerations.size() != 7) {
        ROS_WARN("Joint reset needs q/v/a");
        return;
      }
    for (int j = 0; j < 7; ++j) {
      a.p[j] = p0.positions[order[j]];
      a.v[j] = p0.velocities[order[j]];
      a.a[j] = p0.accelerations[order[j]];
      b.p[j] = p1.positions[order[j]];
      b.v[j] = p1.velocities[order[j]];
      b.a[j] = p1.accelerations[order[j]];
    }
    double T = p1.time_from_start.toSec();
    if (p0.time_from_start.toSec() != 0. || !std::isfinite(T) || T <= 0 ||
        T > 20 || !a.p.allFinite() || !b.p.allFinite() || !a.v.allFinite() ||
        !b.v.allFinite() || !a.a.allFinite() || !b.a.allFinite() ||
        a.v.norm() > 1e-8 || b.v.norm() > 1e-8 || a.a.norm() > 1e-8 ||
        b.a.norm() > 1e-8) {
      ROS_WARN("Invalid joint reset boundaries");
      return;
    }
    if ((a.p.array() < lower_.array()).any() ||
        (a.p.array() > upper_.array()).any() ||
        (b.p.array() < lower_.array()).any() ||
        (b.p.array() > upper_.array()).any()) {
      ROS_WARN("Joint reset outside position limits");
      return;
    }
    Command cmd;
    cmd.valid = true;
    cmd.joint = true;
    cmd.id = msg->header.seq;
    cmd.duration = T;
    cmd.joint_trajectory = Quintic<7>::Fit(a, b, T);
    const auto coefficients = Bernstein(cmd.joint_trajectory, 1);
    for (int j = 0; j < 5; ++j)
      if ((coefficients.col(j).cwiseAbs().array() > velocity_limit_.array())
              .any()) {
        ROS_WARN("Joint reset exceeds continuous velocity bound");
        return;
      }
    cmd.generation = ++generation_;
    buffer_.writeFromNonRT(cmd);
  }
  void Geometry(const std::array<double, 7>& q, const franka::RobotState& state,
                V3* p, V3* axis, J3* J, J3* Jw) {
    const auto pose =
        model_->getPose(franka::Frame::kFlange, q, state.F_T_EE, state.EE_T_K);
    Eigen::Map<const Eigen::Matrix4d> X(pose.data());
    const auto jac = model_->getZeroJacobian(franka::Frame::kFlange, q,
                                             state.F_T_EE, state.EE_T_K);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> js(jac.data());
    const V3 r = X.topLeftCorner<3, 3>() * offset_;
    *p = X.topRightCorner<3, 1>() + r;
    *axis = X.topLeftCorner<3, 3>().col(2);
    *Jw = js.bottomRows<3>();
    *J = ShiftJacobian(r, js.topRows<3>(), *Jw);
  }
  Model ModelAt(const franka::RobotState& state, const V7& velocity) {
    Model m;
    Geometry(state.q, state, &m.p, &m.axis, &m.J, &m.Jw);
    auto mass = model_->getMass();
    auto c = model_->getCoriolis();
    m.M = Eigen::Map<const M7>(mass.data());
    m.coriolis = Eigen::Map<const V7>(c.data());
    std::array<double, 7> plus = state.q, minus = state.q;
    for (int j = 0; j < 7; ++j) {
      plus[j] += 1e-4 * velocity[j];
      minus[j] -= 1e-4 * velocity[j];
    }
    V3 p, z;
    J3 jp, jm, jw;
    Geometry(plus, state, &p, &z, &jp, &jw);
    Geometry(minus, state, &p, &z, &jm, &jw);
    m.bias = (jp - jm) / (2e-4) * velocity;
    return m;
  }
  void PublishTelemetry(const ros::TimerEvent&) {
    const int status = admission_status_.exchange(0);
    if (status)
      ROS_WARN_STREAM("Cartesian command "
                      << admission_id_.load() << " rejected: "
                      << (status == 1   ? "controller busy or faulted"
                          : status == 2 ? "start position mismatch"
                                        : "robot not at rest"));
    std_msgs::Float64MultiArray message;
    uint64_t r = read_.load(std::memory_order_relaxed),
             w = write_.load(std::memory_order_acquire);
    size_t n = std::min<uint64_t>(64, w - r);
    if (!n) return;
    message.layout.dim.resize(2);
    message.layout.dim[0].label = "samples";
    message.layout.dim[0].size = n;
    message.layout.dim[0].stride = n * 51;
    message.layout.dim[1].label =
        "stamp,elapsed,id,q1..7,dq1..7,filtered_dq1..7,p_ref_xyz,v_ref_xyz,p_"
        "xyz,v_xyz,tau_requested1..7,tau_applied1..7,flags";
    message.layout.dim[1].size = 51;
    message.layout.dim[1].stride = 51;
    message.data.reserve(n * 51);
    for (size_t i = 0; i < n; ++i) {
      const auto& s = samples_[(r + i) % capacity];
      message.data.insert(message.data.end(), s.x.begin(), s.x.end());
    }
    read_.store(r + n, std::memory_order_release);
    telemetry_pub_.publish(message);
  }

 public:
  bool init(hardware_interface::RobotHW* hw, ros::NodeHandle& nh) override {
    auto* mi = hw->get<franka_hw::FrankaModelInterface>();
    auto* si = hw->get<franka_hw::FrankaStateInterface>();
    auto* ji = hw->get<hardware_interface::EffortJointInterface>();
    if (!mi || !si || !ji) return false;
    try {
      model_.reset(
          new franka_hw::FrankaModelHandle(mi->getHandle("panda_model")));
      state_.reset(
          new franka_hw::FrankaStateHandle(si->getHandle("panda_robot")));
      for (int j = 0; j < 7; ++j)
        joints_[j] = ji->getHandle("panda_joint" + std::to_string(j + 1));
    } catch (const std::exception& e) {
      ROS_ERROR_STREAM(e.what());
      return false;
    }
    std::vector<double> offset, nominal;
    if (!nh.getParam("ee_offset", offset) || offset.size() != 3) return false;
    offset_ = Eigen::Map<V3>(offset.data());
    if (!offset_.allFinite()) return false;
    has_nominal_ = nh.getParam("q_nominal", nominal);
    if (has_nominal_) {
      if (nominal.size() != 7) return false;
      configured_nominal_ = Eigen::Map<V7>(nominal.data());
      if (!configured_nominal_.allFinite()) return false;
    }
    nh.param("wn", gains_.wn, 20.);
    nh.param("zeta", gains_.zeta, 1.);
    nh.param("posture_wn", gains_.posture_wn, 2.);
    nh.param("axis_k", gains_.axis_k, 5.);
    nh.param("axis_d", gains_.axis_d, 1.);
    nh.param("filter_hz", gains_.filter_hz, 100.);
    nh.param("torque_rate", gains_.torque_rate, 700.);
    nh.param("start_position_tolerance", start_position_tol_, .001);
    nh.param("start_joint_tolerance", start_joint_tol_, .05);
    nh.param("rest_velocity_tolerance", rest_tol_, .05);
    for (double value :
         {gains_.wn, gains_.zeta, gains_.filter_hz, gains_.torque_rate,
          start_position_tol_, start_joint_tol_, rest_tol_})
      if (!std::isfinite(value) || value <= 0) return false;
    for (double value : {gains_.posture_wn, gains_.axis_k, gains_.axis_d})
      if (!std::isfinite(value) || value < 0) return false;
    urdf::Model urdf;
    if (!urdf.initParam("/robot_description")) return false;
    for (int j = 0; j < 7; ++j) {
      auto joint = urdf.getJoint("panda_joint" + std::to_string(j + 1));
      if (!joint || !joint->limits) return false;
      lower_[j] = joint->limits->lower;
      upper_[j] = joint->limits->upper;
      velocity_limit_[j] = joint->limits->velocity;
    }
    cart_sub_ =
        nh.subscribe("/cartesian_reference_trajectory", 1,
                     &PandaCartesianTrajectoryController::Cartesian, this);
    joint_sub_ = nh.subscribe("/joint_reset_trajectory", 1,
                              &PandaCartesianTrajectoryController::Joint, this);
    start_pub_.init(nh, "/cartesian_trajectory_start", 1);
    completion_pub_.init(nh, "/trajectory_completion", 1);
    telemetry_pub_ = nh.advertise<std_msgs::Float64MultiArray>(
        "/cartesian_controller_samples", 8);
    telemetry_timer_ = nh.createTimer(
        ros::Duration(.005),
        &PandaCartesianTrajectoryController::PublishTelemetry, this);
    return true;
  }
  void starting(const ros::Time&) override {
    auto state = state_->getRobotState();
    hold_q_ = Eigen::Map<const V7>(state.q.data());
    filtered_ = Eigen::Map<const V7>(state.dq.data());
    nominal_ = has_nominal_ ? configured_nominal_ : hold_q_;
    active_ = Command();
    seen_ = generation_.load();
    holding_ = true;
    fault_ = false;
    elapsed_ = 0.;
    completed_ = false;
    completion_pending_ = true;
    start_pending_ = false;
    rest_time_ = 0.;
  }
  void stopping(const ros::Time&) override {
    holding_ = true;
    active_ = Command();
    if (completion_pub_.trylock()) {
      completion_pub_.msg_.data = false;
      completion_pub_.unlockAndPublish();
    }
  }
  void update(const ros::Time& time, const ros::Duration& period) override {
    const auto state = state_->getRobotState();
    const V7 q = Eigen::Map<const V7>(state.q.data()),
             dq = Eigen::Map<const V7>(state.dq.data());
    const double dt = period.toSec();
    if (!(dt > 0.) || !std::isfinite(dt)) return;
    filtered_ = Filter(dq, filtered_, dt, gains_.filter_hz);
    Model m;
    try {
      m = ModelAt(state, filtered_);
    } catch (...) {
      fault_ = true;
      holding_ = true;
      hold_q_ = q;
      completion_pending_ = true;
    }
    const Command& incoming = *buffer_.readFromRT();
    if (incoming.valid && incoming.generation != seen_) {
      seen_ = incoming.generation;
      const double position_gap =
          incoming.joint ? (incoming.joint_trajectory.Eval(0).p - q).norm()
                         : (incoming.trajectory.Eval(0).p - m.p).norm();
      const double position_tolerance =
          incoming.joint ? start_joint_tol_ : start_position_tol_;
      const bool position_ok = position_gap <= position_tolerance;
      if (CanAccept(holding_, fault_, position_gap, dq, position_tolerance,
                    rest_tol_)) {
        active_ = incoming;
        elapsed_ = 0.;
        started_ = time.toSec();
        holding_ = false;
        completed_ = false;
        completion_pending_ = true;
        start_pending_ = true;
        rest_time_ = 0.;
        initial_axis_ = m.axis;
        nominal_ = has_nominal_ ? configured_nominal_ : q;
        filtered_ = dq;
      } else {
        admission_id_.store(incoming.id);
        admission_status_.store(!holding_ || fault_ ? 1 : !position_ok ? 2 : 3);
      }
    }
    State<3> ref;
    ref.p = m.p;
    V7 requested;
    if ((q.array() < lower_.array()).any() ||
        (q.array() > upper_.array()).any() ||
        (dq.cwiseAbs().array() > velocity_limit_.array()).any() ||
        !q.allFinite() || !dq.allFinite()) {
      if (!fault_) hold_q_ = q;
      fault_ = true;
      holding_ = true;
      completion_pending_ = true;
    }
    if (!active_.valid || fault_) {
      requested = m.M * (gains_.wn * gains_.wn * (hold_q_ - q) -
                         2 * gains_.zeta * gains_.wn * filtered_) +
                  m.coriolis;
    } else if (active_.joint) {
      const auto x = active_.joint_trajectory.Eval(elapsed_);
      requested = m.M * (x.a + gains_.wn * gains_.wn * (x.p - q) +
                         2 * gains_.zeta * gains_.wn * (x.v - filtered_)) +
                  m.coriolis;
    } else {
      ref = active_.trajectory.Eval(elapsed_);
      auto result = Track(
          m, q, filtered_, holding_ ? hold_q_ : nominal_, ref,
          holding_ ? hold_axis_
                   : AxisTarget(initial_axis_, elapsed_, active_.blend),
          gains_, holding_ ? 0. : FinishWeight(elapsed_, active_.duration));
      requested = result.torque;
      if (!result.valid) {
        fault_ = true;
        holding_ = true;
        hold_q_ = q;
        completion_pending_ = true;
        requested = m.coriolis - 2 * gains_.wn * m.M * filtered_;
      }
    }
    if (!requested.allFinite()) {
      fault_ = true;
      holding_ = true;
      hold_q_ = q;
      completion_pending_ = true;
      requested.setZero();
    }
    bool clipped = false;
    const V7 command =
        Limit(requested, Eigen::Map<const V7>(state.tau_J_d.data()), dt,
              gains_.torque_rate, &clipped);
    for (int j = 0; j < 7; ++j) joints_[j].setCommand(command[j]);
    if (start_pending_ && start_pub_.trylock()) {
      start_pub_.msg_.seq = active_.id;
      start_pub_.msg_.stamp.fromSec(started_);
      start_pub_.msg_.frame_id = "panda_link0";
      start_pub_.unlockAndPublish();
      start_pending_ = false;
    }
    if (active_.valid && !fault_ && elapsed_ >= active_.duration) {
      if (!holding_) {
        hold_q_ = q;
        hold_axis_ = m.axis;
      }
      holding_ = true;
      rest_time_ = dq.cwiseAbs().maxCoeff() <= rest_tol_ ? rest_time_ + dt : 0.;
      if (rest_time_ >= .1 && !completed_) {
        completed_ = true;
        completion_pending_ = true;
      }
    }
    if (completion_pending_ && completion_pub_.trylock()) {
      completion_pub_.msg_.data = completed_ && !fault_;
      completion_pub_.unlockAndPublish();
      completion_pending_ = false;
    }
    const bool near_contact =
        active_.valid && !active_.joint &&
        active_.trajectory.contact_time >= 0 &&
        std::abs(elapsed_ - active_.trajectory.contact_time) <= .3;
    if (near_contact || cycles_ % 5 == 0) {
      uint64_t w = write_.load(std::memory_order_relaxed),
               r = read_.load(std::memory_order_acquire);
      if (w - r < capacity) {
        auto& x = samples_[w % capacity].x;
        int k = 0;
        x[k++] = time.toSec();
        x[k++] = elapsed_;
        x[k++] = active_.id;
        for (const V7& v : {q, dq, filtered_})
          for (int j = 0; j < 7; ++j) x[k++] = v[j];
        for (const V3& v : {ref.p, ref.v, m.p, V3(m.J * dq)})
          for (int j = 0; j < 3; ++j) x[k++] = v[j];
        for (const V7& v : {requested, command})
          for (int j = 0; j < 7; ++j) x[k++] = v[j];
        x[k++] =
            (clipped ? 1 : 0) | (fault_ ? 2 : 0) | (dropped_.load() ? 4 : 0);
        write_.store(w + 1, std::memory_order_release);
      } else
        ++dropped_;
    }
    ++cycles_;
    if (active_.valid && !fault_) elapsed_ += dt;
  }
};
}  // namespace franka_torque_controller
PLUGINLIB_EXPORT_CLASS(
    franka_torque_controller::PandaCartesianTrajectoryController,
    controller_interface::ControllerBase)
