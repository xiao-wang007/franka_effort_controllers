#include <panda_mpc/panda_torque_controller.h>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <franka/robot_state.h>

namespace franka_torque_controller 
{

bool MPCTorqueController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) 
{
  // get model interface
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr)
  {
    ROS_ERROR_STREAM("MPCTorqueController: Error getting model interface from hardware.");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle("panda_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("MPCTorqueController: Exception getting model handle: " << ex.what());
    return false;
  }

  // get state interface
  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr)
  {
    ROS_ERROR_STREAM("MPCTorqueController: Error getting state interface from hardware.");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle("panda_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("MPCTorqueController: Exception getting state handle: " << ex.what());
    return false;
  }

  // get joint torque interface
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) 
  {
    ROS_ERROR_STREAM("MPCTorqueController: Error getting effort joint interface from hardware.");
    return false;
  }
  for (int i = 0; i < NUM_JOINTS; ++i) 
  { 
    try {
       joint_handles_.push_back(effort_joint_interface->getHandle(joint_names_[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM("MPCTorqueController: Exception getting joint handles: " << ex.what());
        return false;
    }
  }

  // publish torque for debugging and analysis
  torque_publisher_ .init(node_handle, "/torque_comparison", 1); //queue size 1

  // init dq_filtered_
  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0.0);

  return true;
}

//########################################################################################
void MPCTorqueController::starting(const ros::Time& time) 
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
  ROS_INFO("MPCTorqueController: Starting controller.");
}

//########################################################################################
void MPCTorqueController::update(const ros::Time& time, const ros::Duration& period) 
{
  if (!u_cmd_received_) 
  {
    ROS_WARN_THROTTLE(1.0, "MPCTorqueController: Waiting for upsampled_u_cmd from solver.");
  }

  if (u_cmd_received_)
  {
    auto torque_cmd = u_cmd_buffer_.readFromRT(); 
    Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> tau_d_calculated(torque_cmd->data());

    franka::RobotState robot_state = state_handle_->getRobotState();
    Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> tau_J_d(robot_state.tau_J.data());

    Eigen::VectorXd tau_cmd = this->SaturateTorqueRate(tau_d_calculated, tau_J_d);

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

}

//########################################################################################
void MPCTorqueController::u_cmd_callback(const std_msgs::Float64MultiArray::ConstPtr& msg) 
{
  if (msg->data.size() != NUM_JOINTS) 
  {
    ROS_ERROR_STREAM("MPCTorqueController: Received u_cmd with incorrect size: " << msg->data.size());
    return;
  }

  // Convert the incoming message to an Eigen vector
  std::array<double, NUM_JOINTS> u_cmd;
  std::copy_n(msg->data.begin(), NUM_JOINTS, u_cmd.begin());

  // Write the command to the realtime buffer
  u_cmd_buffer_.writeFromNonRT(u_cmd);
  u_cmd_received_ = true;
}

//########################################################################################
Eigen::Matrix<double, 7, 1> MPCTorqueController::SaturateTorqueRate(
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

PLUGINLIB_EXPORT_CLASS(franka_torque_controller::MPCTorqueController, controller_interface::ControllerBase)