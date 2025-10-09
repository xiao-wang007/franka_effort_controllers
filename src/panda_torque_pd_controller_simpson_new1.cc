#include <panda_mpc/panda_torque_pd_controller_simpson.h>

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

  // init starting time publisher
  start_time_publisher_ = node_handle.advertise<std_msgs::Float64>("/controller_t_start",
                                                              1, true); // queue size 1, latched

  // init trajectory completion publisher
  traj_completion_pub_ = node_handle.advertise<std_msgs::Bool>("/trajectory_completion", 1, true); // latched

  return true;
}

//########################################################################################
void TorquePDController_Simpson::starting(const ros::Time& time) 
{
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

  // load the ref trajs
  int nJoint = 7;
  
  /* for earlier ones */
  // auto loaded_q = load_csv(ref_traj_path_q_, N_, nJoint);
  // auto loaded_v = load_csv(ref_traj_path_v_, N_, nJoint);
  // auto loaded_u = load_csv(ref_traj_path_u_, N_, nJoint);
  // auto loaded_h = load_csv(ref_traj_path_h_, N_-1, 1);
  // auto loaded_a = load_csv(ref_traj_path_a_, N_, nJoint);

  /* for later ones from effective mass onwards*/
  auto loaded_q = load_csv(ref_traj_path_q_, N_, nJoint);
  auto loaded_v = load_csv(ref_traj_path_v_, N_, nJoint);
  auto loaded_u = load_csv(ref_traj_path_u_, N_, nJoint);
  auto loaded_h = load_csv(ref_traj_path_h_, N_, 1);
  auto loaded_a = load_csv(ref_traj_path_a_, N_, nJoint);

  std::cout << "loaded_q shape: " << loaded_q.rows() << " x " << loaded_q.cols() << std::endl;
  std::cout << "loaded_v shape: " << loaded_v.rows() << " x " << loaded_v.cols() << std::endl;
  std::cout << "loaded_u shape: " << loaded_u.rows() << " x " << loaded_u.cols() << std::endl;
  std::cout << "loaded_h shape: " << loaded_h.rows() << " x " << loaded_h.cols() << std::endl;
  std::cout << "loaded_a shape: " << loaded_a.rows() << " x " << loaded_a.cols() << std::endl;

  // compute time knots 
  std::vector<double> ts;
  /* for earlier ones */
  // auto cumsum_h = cumulative_sum(loaded_h);
  // std::cout << "cumsum_h: " << cumsum_h.transpose() << '\n' << std::endl;
  // ts.push_back(0.0); // start from 0 second
  // for (int i = 0; i < cumsum_h.rows(); i++)
  // {
  //   ts.push_back(cumsum_h(i));
  // }

  for (int i = 0; i < loaded_h.rows(); i++)
  {
    ts.push_back(loaded_h(i));
  }

  // Convert std::vector<double> to Eigen::VectorXd
  Eigen::VectorXd ts_eigen = Eigen::Map<Eigen::VectorXd>(ts.data(), ts.size());
  q_hermite_spline_.fit(ts_eigen, loaded_q, loaded_v);
  v_hermite_spline_.fit(ts_eigen, loaded_v, loaded_a);
  u_quadratic_spline_.fit(ts_eigen, loaded_u);

  // linear spline for u
  std::vector<Vec7> us;
  for (int i = 0; i < loaded_u.rows(); i++)
  {
      Vec7 u = loaded_u.row(i).transpose();
      us.push_back(u);
  }
  u_linear_spline_.reset(ts, us);


  // set traj end time
  traj_completion_time_ = ts.back() + t_delay_;
  ROS_INFO("Trajectory end time (with delay = 0.1s): %.3f seconds \n", traj_completion_time_);

  /* signal when restarting the controller */
  std_msgs::Bool msg;
  msg.data = false;
  traj_completion_pub_.publish(msg);

  // signal to console which case it is
  std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%% " << message_to_console_ << " %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;

  std::cout << '\n' << "Controller gains: " << std::endl;
  std::cout << "Kp: " << Kp_.transpose() << std::endl;
  std::cout << "Kd: " << Kd_.transpose() << '\n' << std::endl;

  std::cout << "q_now: " << q_now_.transpose() << '\n' << std::endl;

  if (use_t_varying_gains_) 
  {
    std::cout << "Damping ratio zeta: " << zeta_ << std::endl;
    std::cout << "Natural frequency wn: " << wn_ << "\n" << std::endl;  
  } 

  // set controller start time
  t_traj_ = 0.0; 

  // publish the starting time
  std_msgs::Float64 t_start_msg;
  t_start_msg.data = time.toSec();
  start_time_publisher_.publish(t_start_msg);

}

//########################################################################################
void TorquePDController_Simpson::update(const ros::Time& time, const ros::Duration& period) 
{
  // get time
  t_traj_ += period.toSec();

  // get current state
  franka::RobotState robot_state = state_handle_->getRobotState();
  Eigen::Map<const Eigen::Matrix<double,7,1>> tau_J_d(robot_state.tau_J_d.data());

  /* get external torque from build-in interface */ 
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_ext(robot_state.tau_ext_hat_filtered.data());
  if (tau_ext.norm() > 2.0)
  {
    std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    std::cout << "%%%%%%% Contact detected!" << t_traj_ << " %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;
  }

  /* manually estimate external torque */ 
  

  // index to get desired q, v, and tau_ff
  Eigen::VectorXd q_d = q_hermite_spline_.eval(t_traj_);
  Eigen::VectorXd v_d = v_hermite_spline_.eval(t_traj_);
  // Eigen::VectorXd tau_ff_quadratic = u_quadratic_spline_.eval(t_traj_); // overshoots between knot points, do not use
  Eigen::VectorXd tau_ff_linear = u_linear_spline_(t_traj_);

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

  // Check if trajectory is complete based on time
  if (t_traj_ >= traj_completion_time_) 
  {
    std_msgs::Bool msg;
    msg.data = true;
    traj_completion_pub_.publish(msg);
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
  ROS_INFO("TorquePDController_Simpson_with_detection: Stopping controller, switching to another controller.");
}

//########################################################################################
bool TorquePDController_Simpson::loadParameters(ros::NodeHandle& node_handle) 
{
  bool params_loaded = true;
  
  // Load trajectory file paths - all are required
  if (!node_handle.getParam("ref_traj_path_h", ref_traj_path_h_)) {
    ROS_ERROR("TorquePDController_Simpson: Required parameter 'ref_traj_path_h' not found!");
    params_loaded = false;
  }
  
  if (!node_handle.getParam("ref_traj_path_q", ref_traj_path_q_)) {
    ROS_ERROR("TorquePDController_Simpson: Required parameter 'ref_traj_path_q' not found!");
    params_loaded = false;
  }
  
  if (!node_handle.getParam("ref_traj_path_v", ref_traj_path_v_)) {
    ROS_ERROR("TorquePDController_Simpson: Required parameter 'ref_traj_path_v' not found!");
    params_loaded = false;
  }
  
  if (!node_handle.getParam("ref_traj_path_u", ref_traj_path_u_)) {
    ROS_ERROR("TorquePDController_Simpson: Required parameter 'ref_traj_path_u' not found!");
    params_loaded = false;
  }
  
  if (!node_handle.getParam("ref_traj_path_a", ref_traj_path_a_)) {
    ROS_ERROR("TorquePDController_Simpson: Required parameter 'ref_traj_path_a' not found!");
    params_loaded = false;
  }
  
  // Return early if required trajectory parameters are missing
  if (!params_loaded) {
    ROS_ERROR("TorquePDController_Simpson: Controller initialization failed due to missing required trajectory parameters!");
    return false;
  }
  
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
                  << "ref_traj_path_h: " << ref_traj_path_h_ << "\n"
                  << "ref_traj_path_q: " << ref_traj_path_q_ << "\n" 
                  << "ref_traj_path_v: " << ref_traj_path_v_ << "\n"
                  << "ref_traj_path_u: " << ref_traj_path_u_ << "\n"
                  << "ref_traj_path_a: " << ref_traj_path_a_ << "\n"
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

} // namespace franka_torque_controller

PLUGINLIB_EXPORT_CLASS(franka_torque_controller::TorquePDController_Simpson, controller_interface::ControllerBase)