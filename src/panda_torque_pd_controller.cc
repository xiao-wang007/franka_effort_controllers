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

  // publish torque for debugging and analysis
  torque_publisher_.init(node_handle, "/torque_comparison", 1); //queue size 1

  // init dq_filtered_
  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0.0);

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
  int N = 60;
  int nJoint = 7;
  auto loaded_q = load_csv(ref_traj_path_q_, N, nJoint);
  auto loaded_v = load_csv(ref_traj_path_v_, N, nJoint);
  auto loaded_u = load_csv(ref_traj_path_u_, N, nJoint);
  auto loaded_h = load_csv(ref_traj_path_h_, N-1, 1);


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
  std::cout << "u_spline(0.12): " << u_spline_(0.12).transpose() << std::endl;

  // override the kp gain for joint 7 
  //Kp_(6) = 10.;
  Kd_(6) = 5.; // lower the kd gain for joint 7 kills the jittering
  std::cout << "Kp: " << Kp_.transpose() << std::endl;
  std::cout << "Kd: " << Kd_.transpose() << '\n' << std::endl;

  // get controller start time
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


} // namespace franka_torque_controller

PLUGINLIB_EXPORT_CLASS(franka_torque_controller::TorquePDController, controller_interface::ControllerBase)