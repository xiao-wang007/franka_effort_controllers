#include <panda_mpc/panda_torque_pd_controller_simpson.h>

#include <algorithm>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <franka/robot_state.h>

namespace franka_torque_controller 
{

bool TorquePDController_Simpson::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) 
{
  // get model interface
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr)
  {
    ROS_ERROR_STREAM("TorquePDController: Error getting model interface from hardware.");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle("panda_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("TorquePDController: Exception getting model handle: " << ex.what());
    return false;
  }

  // get state interface
  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr)
  {
    ROS_ERROR_STREAM("TorquePDController: Error getting state interface from hardware.");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle("panda_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("TorquePDController: Exception getting state handle: " << ex.what());
    return false;
  }

  // get joint torque interface
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) 
  {
    ROS_ERROR_STREAM("TorquePDController: Error getting effort joint interface from hardware.");
    return false;
  }
  for (int i = 0; i < NUM_JOINTS; ++i) 
  { 
    try {
       joint_handles_.push_back(effort_joint_interface->getHandle(joint_names_[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM("TorquePDController: Exception getting joint handles: " << ex.what());
        return false;
    }
  }

  // init dq_filtered_
  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0.0);

  // Load parameters from YAML file
  if (!loadParameters(node_handle)) {
    ROS_ERROR("TorquePDController_Simpson: Failed to load parameters from YAML file.");
    return false;
  }

  // publish torque for debugging and analysis
  torque_publisher_.init(node_handle, "/torque_comparison", 1); //queue size 1

  // subscribe to the solved trajectory, published as a
  // trajectory_msgs/JointTrajectory (see tools/send_trajectory.py in drakecpp)
  trajectory_subscriber_ = node_handle.subscribe(
      "/reference_trajectory", 1,
      &TorquePDController_Simpson::trajectoryCallback, this);

  // init starting time publisher
  start_time_publisher_ = node_handle.advertise<std_msgs::Float64>("/controller_t_start",
                                                              1, true); // queue size 1, latched

  // init trajectory completion publisher
  traj_completion_pub_ = node_handle.advertise<std_msgs::Bool>("/trajectory_completion", 1, true); // latched

  return true;
}

//########################################################################################
bool TorquePDController_Simpson::loadParameters(ros::NodeHandle& node_handle) 
{
  // Load controller gains
  std::vector<double> kp_gains_vec, kd_gains_vec;
  if (node_handle.getParam("kp_gains", kp_gains_vec) && kp_gains_vec.size() == NUM_JOINTS) {
    Kp_ = Eigen::Map<Eigen::VectorXd>(kp_gains_vec.data(), NUM_JOINTS);
  } else {
    ROS_WARN("TorquePDController_Simpson: kp_gains parameter not found or wrong size, using default");
    Kp_ = Eigen::VectorXd::Constant(NUM_JOINTS, 40.0);
  }
  
  if (node_handle.getParam("kd_gains", kd_gains_vec) && kd_gains_vec.size() == NUM_JOINTS) {
    Kd_ = Eigen::Map<Eigen::VectorXd>(kd_gains_vec.data(), NUM_JOINTS);
  } else {
    ROS_WARN("TorquePDController_Simpson: kd_gains parameter not found or wrong size, using default");
    Kd_ = Eigen::VectorXd::Constant(NUM_JOINTS, 40.0);
  }
  
  // Load other parameters
  if (!node_handle.getParam("alpha", alpha_)) {
    ROS_WARN("TorquePDController_Simpson: alpha parameter not found, using default");
    alpha_ = 0.99;
  }
  
  if (!node_handle.getParam("N", N_)) {
    ROS_WARN("TorquePDController_Simpson: N parameter not found, using default");
    N_ = 20;
  }
  
  if (!node_handle.getParam("message_to_console", message_to_console_)) {
    ROS_WARN("TorquePDController_Simpson: message_to_console parameter not found, using default");
    message_to_console_ = "Tracking with Simpson's, N = " + std::to_string(N_);
  }

  if (!node_handle.getParam("use_t_varying_gains", use_t_varying_gains_)) {
    ROS_WARN("TorquePDController_Simpson: use_t_varying_gains parameter not found, using default");
    use_t_varying_gains_ = false;
  }

  if (!node_handle.getParam("zeta", zeta_)) {
    ROS_WARN("TorquePDController_Simpson: zeta parameter not found, using default");
    zeta_ = 0.7;
  }

  if (!node_handle.getParam("natural_frequency", wn_)) {
    ROS_WARN("TorquePDController_Simpson: natural frequency parameter not found, using default");
    wn_ = 2.0 * 3.14 * 4.0; // 4 Hz bandwidth
  }

  
  // Log loaded parameters
  ROS_INFO_STREAM("TorquePDController_Simpson: Loaded parameters:\n"
                  << "Kp gains: " << Kp_.transpose() << "\n"
                  << "Kd gains: " << Kd_.transpose() << "\n"
                  << "alpha: " << alpha_ << "\n"
                  << "N: " << N_ << "\n"
                  << "message: " << message_to_console_ << "\n"
                  << "use_t_varying_gains: " << (use_t_varying_gains_ ? "true" : "false") << "\n"
                  << "zeta: " << zeta_ << "\n"
                  << "natural_frequency: " << wn_ << "\n");

  return true;
}

//########################################################################################
void TorquePDController_Simpson::trajectoryCallback(
    const trajectory_msgs::JointTrajectory::ConstPtr& msg)
{
  const int n = static_cast<int>(msg->points.size());
  if (n < 2) {
    ROS_WARN("TorquePDController_Simpson: received trajectory with < 2 points, ignoring");
    return;
  }

  // Map the incoming joint order onto joint_names_'s order, in case a
  // producer sends them in a different order.
  std::array<int, NUM_JOINTS> col{};
  for (int i = 0; i < NUM_JOINTS; ++i) {
    const auto it = std::find(msg->joint_names.begin(), msg->joint_names.end(), joint_names_[i]);
    if (it == msg->joint_names.end()) {
      ROS_ERROR_STREAM("TorquePDController_Simpson: trajectory is missing joint "
                       << joint_names_[i] << ", ignoring");
      return;
    }
    col[i] = static_cast<int>(std::distance(msg->joint_names.begin(), it));
  }

  Eigen::VectorXd ts(n);
  Eigen::MatrixXd q(n, NUM_JOINTS), v(n, NUM_JOINTS), u(n, NUM_JOINTS);
  for (int i = 0; i < n; ++i) {
    const auto& pt = msg->points[i];
    ts(i) = pt.time_from_start.toSec();
    for (int j = 0; j < NUM_JOINTS; ++j) {
      q(i, j) = pt.positions.at(col[j]);
      v(i, j) = pt.velocities.at(col[j]);
      u(i, j) = pt.effort.empty() ? 0.0 : pt.effort.at(col[j]);
    }
  }
  for (int i = 1; i < n; ++i) {
    if (!(ts(i) > ts(i - 1))) {
      ROS_ERROR("TorquePDController_Simpson: trajectory times_from_start are not "
                "strictly increasing, ignoring");
      return;
    }
  }

  // Acceleration at each knot, needed to fit v's Hermite spline: central
  // differences of velocity, one-sided at the endpoints.
  Eigen::MatrixXd a(n, NUM_JOINTS);
  a.row(0) = (v.row(1) - v.row(0)) / (ts(1) - ts(0));
  a.row(n - 1) = (v.row(n - 1) - v.row(n - 2)) / (ts(n - 1) - ts(n - 2));
  for (int i = 1; i < n - 1; ++i) {
    a.row(i) = (v.row(i + 1) - v.row(i - 1)) / (ts(i + 1) - ts(i - 1));
  }

  auto data = std::make_shared<TrajectoryData>();
  data->q_spline.fit(ts, q, v);
  data->v_spline.fit(ts, v, a);
  std::vector<double> ts_vec(ts.data(), ts.data() + ts.size());
  std::vector<Vec7> us;
  us.reserve(n);
  for (int i = 0; i < n; ++i) us.push_back(u.row(i).transpose());
  data->u_spline.reset(ts_vec, us);
  data->duration = ts(n - 1);

  trajectory_buffer_.writeFromNonRT(data);
  ROS_INFO_STREAM("TorquePDController_Simpson: received new trajectory, "
                  << n << " points, duration " << data->duration << " s");
}

//########################################################################################
void TorquePDController_Simpson::starting(const ros::Time& time) 
{
  // Reset completion flags
  traj_completion_published_ = false;
  trajectory_finished_ = false;

  franka::RobotState robot_state = state_handle_->getRobotState();

  // map to eigen for printing
  Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> q_now_(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> v_now_(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> u_now_(robot_state.tau_J.data());
  
  // Initialize the controller state
  ROS_INFO_STREAM("Current robot state: \n"
        << "q_now: " << q_now_.transpose() << "\n"
        << "v_now: " << v_now_.transpose() << "\n"
        << "u_now: " << u_now_.transpose() << "\n");
  ROS_INFO("TorquePDController: Starting controller.");


  // pick up whatever trajectory has been received so far on /reference_trajectory
  active_trajectory_ = *trajectory_buffer_.readFromRT();
  if (!active_trajectory_) {
    ROS_ERROR("TorquePDController_Simpson: starting with no trajectory received yet; "
              "publish one on /reference_trajectory before (or after) starting this "
              "controller -- update() will hold zero torque until one arrives.");
  } else {
    ROS_INFO_STREAM("TorquePDController_Simpson: starting with trajectory duration "
                    << active_trajectory_->duration << " s");
  }

  /* for N = 20, tried with simpson */
  // Kp_.resize(NUM_JOINTS);
  // Kp_ << 80., 80., 80., 80., 60., 80., 20.;
  // Kd_.resize(NUM_JOINTS);
  // Kd_ << 70., 70., 70., 70., 10., 10., 5.;

  /* for N = 60, tried with simpson */
  // Kp_.resize(NUM_JOINTS);
  // Kp_ << 40., 40., 40., 40., 40., 40., 40.;
  // Kd_.resize(NUM_JOINTS);
  // Kd_ << 30., 30., 30., 30., 10., 10., 5.;

  /* for case 6, meff large rotation*/
  // Kp_.resize(NUM_JOINTS);
  // Kp_ << 60., 120., 60., 100., 60., 60., 60.;
  // Kd_.resize(NUM_JOINTS);
  // Kd_ << 40., 80., 40., 80., 20., 20., 10.;
  // Kp_ << 50., 15., 15., 15., 15., 15., 15.;
  // Kd_ << 5., 5., 5., 5., 5., 5., 5.;

  //Kp_(0) = 50.;
  //Kp_(6) = 10.;
  // Kd_(6) = 5.; // lower the kd gain for joint 7 kills the jittering
  // Kd_(5) = 10.; // lower the kd gain for joint 6 kills the jittering
  // Kd_(4) = 10.; // lower the kd gain for joint 5 kills the jittering

  // publish the starting time
  std_msgs::Float64 t_start_msg;
  t_start_msg.data = time.toSec();
  start_time_publisher_.publish(t_start_msg);

  // set traj end time
  traj_completion_time_ = active_trajectory_ ? active_trajectory_->duration + t_delay_ : 0.0;
  ROS_INFO("Trajectory end time (with delay = 0.1s): %.3f seconds \n", traj_completion_time_);

  /* signal when restarting the controller */
  std_msgs::Bool msg;
  msg.data = false;
  traj_completion_pub_.publish(msg);

  // signal to console which case it is
  std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%% " << message_to_console_ << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;

  std::cout << "q_now: " << q_now_.transpose() << '\n' << std::endl;

  if (use_t_varying_gains_) 
  {
    std::cout << "Damping ratio zeta: " << zeta_ << std::endl;
    std::cout << "Natural frequency wn: " << wn_ << "\n" << std::endl;  
  } 
  else{
    std::cout << '\n' << "Controller gains: " << std::endl;
    std::cout << "Kp: " << Kp_.transpose() << std::endl;
    std::cout << "Kd: " << Kd_.transpose() << '\n' << std::endl;
  }

  // set controller start time
  t_traj_ = 0.0; 
}

//########################################################################################
void TorquePDController_Simpson::update(const ros::Time& time, const ros::Duration& period) 
{
  // Pick up a newly published trajectory as soon as it arrives, so a fresh
  // solve can be tracked without stopping/restarting this controller.
  const auto& latest = *trajectory_buffer_.readFromRT();
  if (latest && latest != active_trajectory_) {
    active_trajectory_ = latest;
    t_traj_ = 0.0;
    traj_completion_time_ = active_trajectory_->duration + t_delay_;
    traj_completion_published_ = false;
    trajectory_finished_ = false; // leaving the hold phase: a fresh trajectory is now active
    ROS_INFO_STREAM("TorquePDController_Simpson: switched to new trajectory, duration "
                    << active_trajectory_->duration << " s");
  }

  // get current state
  franka::RobotState robot_state = state_handle_->getRobotState();
  Eigen::Map<const Eigen::Matrix<double,7,1>> tau_J_d(robot_state.tau_J_d.data());

  if (!active_trajectory_) {
    // No trajectory received yet: hold zero commanded torque (Franka adds its
    // own gravity compensation outside this interface) rather than evaluate
    // splines that don't exist.
    static franka_hw::TriggerRate warn_rate{1.0};
    if (warn_rate()) {
      ROS_WARN("TorquePDController_Simpson: no trajectory received yet, holding zero torque");
    }
    for (int i = 0; i < NUM_JOINTS; ++i) joint_handles_[i].setCommand(0.0);
    return;
  }

  // get time
  t_traj_ += period.toSec();

  // ------------------------------------------------------------------
  // Reference past the trajectory's end is a deliberate HOLD, not a bug
  // in extrapolation: once t_traj_ exceeds duration, q_d, v_d and tau_ff
  // freeze at the final knot. v_d is 0 by construction, and tau_ff is the
  // last knot's effort (0 in our convention -- gravity is excluded from
  // the solve and the FCI adds its own gravity compensation), so the arm
  // parks at the final pose between segments. We clamp the evaluation
  // time to `duration` so this behaviour is stated here explicitly,
  // instead of being hidden inside each spline's zero-order-hold.
  // ------------------------------------------------------------------
  const bool holding_at_end = t_traj_ >= active_trajectory_->duration;
  const double t_ref = holding_at_end ? active_trajectory_->duration : t_traj_;

  // index to get desired q, v, and tau_ff at the (clamped) reference time
  Eigen::VectorXd q_d = active_trajectory_->q_spline.eval(t_ref);
  Eigen::VectorXd v_d = active_trajectory_->v_spline.eval(t_ref);
  Eigen::VectorXd tau_ff_linear = active_trajectory_->u_spline(t_ref);

  // log once when the hold phase begins (flag is reset in starting() and
  // whenever a new trajectory is picked up below)
  if (holding_at_end && !trajectory_finished_) {
    trajectory_finished_ = true;
    ROS_INFO_STREAM("Reference held at final pose -- q_d: " << q_d.transpose()
                    << ", v_d: " << v_d.transpose()
                    << ", tau_ff: " << tau_ff_linear.transpose());
  }


  // filter out joint7 velocity
   for (size_t i = 0; i < 7; i++) {
    dq_filtered_[i] = (1 - alpha_) * dq_filtered_[i] + alpha_ * robot_state.dq[i];
  }

  if (use_t_varying_gains_) 
  {
    // get inertia matrix and map to eigen
    const std::array<double, 49> M = model_handle_->getMass(); 
    Eigen::Map<const Eigen::Matrix<double, NUM_JOINTS, NUM_JOINTS>> M_eigen(M.data());

    // use only diagonal
    Eigen::VectorXd M_diag = M_eigen.diagonal();
    Kp_.resize(NUM_JOINTS);
    Kp_ = wn_ * wn_ * M_diag;
    Kd_.resize(NUM_JOINTS);
    Kd_ = 2.0 * zeta_ * wn_ * M_diag;
  }


  // compute the torque
  Eigen::VectorXd tau_calculated(NUM_JOINTS);
  for(int i=0; i<NUM_JOINTS; i++)
  {
    // tau_calculated(i) = tau_ff(i) 
    //                   + Kp_(i) * (q_d(i) - robot_state.q[i]) 
    //                   + Kd_(i) * (v_d(i) - robot_state.dq[i]);
    tau_calculated(i) = tau_ff_linear(i) 
                      + Kp_(i) * (q_d(i) - robot_state.q[i]) 
                      + Kd_(i) * (v_d(i) - dq_filtered_[i]);
  }

  // clamp the torque to be within limits
  tau_calculated = tau_calculated.cwiseMax(tau_min);
  tau_calculated = tau_calculated.cwiseMin(tau_max);

  // saturate the torque rate
  Eigen::VectorXd tau_cmd = this->SaturateTorqueRate(tau_calculated, tau_J_d);

  for (int i = 0; i < NUM_JOINTS; ++i) 
  {
      joint_handles_[i].setCommand(tau_cmd[i]);
  }

  // Publish the torque command if the trigger rate allows it
  if (trigger_rate_() && torque_publisher_.trylock())
  {
    torque_publisher_.msg_.data.clear(); // clear before push back
    for (size_t i = 0; i < NUM_JOINTS; ++i) 
    {
      torque_publisher_.msg_.data.push_back(tau_cmd[i]);
    }
    torque_publisher_.unlockAndPublish();
  } /* forgot why I did this! Damn. Probably I want to manually compare the torques */

  // Check if trajectory is complete based on time
  if (!traj_completion_published_ && t_traj_ >= traj_completion_time_) 
  {
    std_msgs::Bool msg;
    msg.data = true;
    traj_completion_pub_.publish(msg);
    traj_completion_published_ = true;
    ROS_INFO("Trajectory complete at t=%.3f seconds", t_traj_);
  }
}

//########################################################################################
Eigen::Matrix<double, 7, 1> TorquePDController_Simpson::SaturateTorqueRate(
                            const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
                            const Eigen::Matrix<double, 7, 1>& tau_J_d) 
{  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

//########################################################################################
void TorquePDController_Simpson::stopping(const ros::Time& /*time*/) 
{
  
    std_msgs::Bool msg;
    msg.data = false;
    traj_completion_pub_.publish(msg);
  
  ROS_INFO("TorquePDController: Stopping controller, reset completion status.");
}

} // namespace franka_torque_controller

PLUGINLIB_EXPORT_CLASS(franka_torque_controller::TorquePDController_Simpson, controller_interface::ControllerBase)