#include <panda_mpc/panda_torque_pd_controller.h>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <franka/robot_state.h>

namespace franka_torque_controller 
{

bool TorquePDController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) 
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

  // publish torque for debugging and analysis
  torque_publisher_.init(node_handle, "/torque_comparison", 1); //queue size 1

  // init starting time publisher
  start_time_publisher_ = node_handle.advertise<std_msgs::Float64>("/controller_t_start",
                                                              1, true); // queue size 1, latched

  // init trajectory completion publisher
  traj_completion_pub_ = node_handle.advertise<std_msgs::Bool>("/trajectory_completion", 1, true); // latched

  return true;
}

//########################################################################################
void TorquePDController::starting(const ros::Time& time) 
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
  auto loaded_q = load_csv(ref_traj_path_q_, N_, nJoint);
  auto loaded_v = load_csv(ref_traj_path_v_, N_, nJoint);
  auto loaded_u = load_csv(ref_traj_path_u_, N_, nJoint);
  auto loaded_h = load_csv(ref_traj_path_h_, N_-1, 1);


  // compute time knots 
  std::vector<double> ts;
  auto cumsum_h = cumulative_sum(loaded_h);
  std::cout << "cumsum_h: " << cumsum_h.transpose() << '\n' << std::endl;
  ts.push_back(0.0); // start from 0 second
  for (int i = 0; i < cumsum_h.rows(); i++)
  {
    ts.push_back(cumsum_h(i));
  }

  // linear spline for q 
  std::vector<Vec7> qs;
  for (int i = 0; i < loaded_q.rows(); i++)
  {
      Vec7 q = loaded_q.row(i).transpose();
      qs.push_back(q);
  }
  q_spline_.reset(ts, qs);
  std::cout << "q_spline(0.0): " << q_spline_(0.0).transpose() << std::endl;
  std::cout << "q_spline(0.12): " << q_spline_(0.12).transpose() << '\n' << std::endl;

  // linear spline for v 
  std::vector<Vec7> vs;
  for (int i = 0; i < loaded_v.rows(); i++)
  {
      Vec7 v = loaded_v.row(i).transpose();
      vs.push_back(v);
  }
  v_spline_.reset(ts, vs);
  std::cout << "v_spline(0.0): " << v_spline_(0.0).transpose() << std::endl;
  std::cout << "v_spline(0.12): " << v_spline_(0.12).transpose() << '\n' << std::endl;

  // linear spline for u
  std::vector<Vec7> us;
  for (int i = 0; i < loaded_u.rows(); i++)
  {
      Vec7 u = loaded_u.row(i).transpose();
      us.push_back(u);
  }
  u_spline_.reset(ts, us);
  std::cout << "u_spline(0.0): " << u_spline_(0.0).transpose() << std::endl;
  std::cout << "u_spline(0.12): " << u_spline_(0.12).transpose() << '\n' << std::endl;

  /* this set is the same as used in N = 20 */
  // Kp_.resize(NUM_JOINTS);
  // Kp_ << 40., 40., 40., 40., 40., 40., 40.;
  // Kd_.resize(NUM_JOINTS);
  // Kd_ << 30., 30., 30., 30., 10., 10., 5.;

  /* These are used for N = 60 Euler only, simpson needs large gains */
  // // override the kp gain for joint 7 
  // Kp_(0) = 50.;
  // //Kp_(6) = 10.;
  // Kd_(6) = 5.; // lower the kd gain for joint 7 kills the jittering
  Kp_.resize(NUM_JOINTS);
  Kp_ << 150., 150., 150., 150., 150., 150., 150.;
  Kd_.resize(NUM_JOINTS);
  Kd_ << 30., 30., 30., 30., 30., 30., 5.;
  std::cout << "Kp: " << Kp_.transpose() << std::endl;
  std::cout << "Kd: " << Kd_.transpose() << '\n' << std::endl;

  // publish the starting time
  std_msgs::Float64 t_start_msg;
  t_start_msg.data = time.toSec();
  start_time_publisher_.publish(t_start_msg);

  // set traj end time
  traj_completion_time_ = ts.back() + t_delay_;
  ROS_INFO("Trajectory end time (with delay = 0.1s): %.3f seconds \n", traj_completion_time_);
  traj_completion_published_ = false;

  // signal to console which case is tracking
  std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%% " << message_to_console_ << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;

  // set controller start time
  t_traj_ = 0.0; 
}

//########################################################################################
void TorquePDController::update(const ros::Time& time, const ros::Duration& period) 
{
  // get current state
  franka::RobotState robot_state = state_handle_->getRobotState();
  Eigen::Map<const Eigen::Matrix<double,7,1>> tau_J_d(robot_state.tau_J_d.data());

  // get time
  t_traj_ += period.toSec();

  // index to get desired q, v, and tau_ff
  Eigen::VectorXd q_d = q_spline_(t_traj_);
  Eigen::VectorXd v_d = v_spline_(t_traj_);
  Eigen::VectorXd tau_ff = u_spline_(t_traj_);

  // filter out joint7 velocity
   for (size_t i = 0; i < 7; i++) {
    dq_filtered_[i] = (1 - alpha_) * dq_filtered_[i] + alpha_ * robot_state.dq[i];
  }

  // compute the torque
  Eigen::VectorXd tau_calculated(NUM_JOINTS);
  for(int i=0; i<NUM_JOINTS; i++)
  {
    // tau_calculated(i) = tau_ff(i) 
    //                   + Kp_(i) * (q_d(i) - robot_state.q[i]) 
    //                   + Kd_(i) * (v_d(i) - robot_state.dq[i]);
    tau_calculated(i) = tau_ff(i) 
                      + Kp_(i) * (q_d(i) - robot_state.q[i]) 
                      + Kd_(i) * (v_d(i) - dq_filtered_[i]);
  }

  // saturate the torque rate
  Eigen::VectorXd tau_cmd = this->SaturateTorqueRate(tau_calculated, tau_J_d);

  for (int i = 0; i < NUM_JOINTS; ++i) 
  {
      joint_handles_[i].setCommand(tau_cmd[i]);
  }

  // Publish the torque command if the trigger rate allows it
  if (trigger_rate_() && torque_publisher_.trylock())
  {
    for (size_t i = 0; i < NUM_JOINTS; ++i) 
    {
      torque_publisher_.msg_.data.push_back(tau_cmd[i]);
    }
  } /* forgot why I did this! Damn. */

  // Check if trajectory is complete based on time
  if (!traj_completion_published_ && t_traj_ >= traj_completion_time_) 
  {
    std_msgs::Bool msg;
    msg.data = true;
    traj_completion_pub_.publish(msg);
    traj_completion_published_ = true;
    ROS_INFO("Trajectory complete at t=%.3f seconds", t_traj_);

    // Don't shut down publishers, just set a flag
    trajectory_finished_ = true;
  }
}

//########################################################################################
Eigen::Matrix<double, 7, 1> TorquePDController::SaturateTorqueRate(
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
void TorquePDController::stopping(const ros::Time& /*time*/) 
{
  // Reset completion flags
  traj_completion_published_ = false;
  trajectory_finished_ = false;
  
  // Optional: Reset other state that should be initialized when restarting
  t_traj_ = 0.0;
  
  // Only publish if we need to change the completion status
  if (traj_completion_pub_.getNumSubscribers() > 0) {
    std_msgs::Bool msg;
    msg.data = false;
    traj_completion_pub_.publish(msg);
  }
  
  ROS_INFO("TorquePDController: Stopping controller, reset completion status.");
}

} // namespace franka_torque_controller

PLUGINLIB_EXPORT_CLASS(franka_torque_controller::TorquePDController, controller_interface::ControllerBase)