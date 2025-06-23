#pragma once
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>

//#include <franka_example_controllers/JointTorqueComparison.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/trigger_rate.h>

namespace franka_torque_controller {
 
#define NUM_JOINTS 7
#define PI 3.14

class MPCTorqueController : public controller_interface::MultiInterfaceController<
                                    franka_hw::FrankaModelInterface,
                                    franka_hw::FrankaStateInterface,
                                    hardware_interface::EffortJointInterface> 
{
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;

  void update(const ros::Time& time, const ros::Duration& period) override;

  void starting(const ros::Time& time) override;

  //void stopping(const ros::Time& time) override;

 private:
  std::array<double, 7> saturateTorqueRate(const std::array<double, 7>& tau_d_calculated,
                                           const std::array<double, 7>& tau_J_d);

  void u_cmd_callback(const std_msgs::Float64MultiArray::ConstPtr& msg);
  

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  franka_hw::TriggerRate trigger_rate_ {0.1}; // 10 Hz
  static constexpr double kMaxTorqueRate = 0.1; 
  const double delta_tau_max_ {1.0}; 
  std::array<double, 7> last_tau_d_ {};
  std::array<double, 7> dq_filtered_ {};
  realtime_tools::RealtimeBuffer<std::array<double, 7>> u_cmd_buffer_;
  std::atomic<bool> u_cmd_received_ {false};

  // sub and pub 
  ros::Subscriber u_cmd_subscriber_;
  realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray> torque_publisher_;
}
}