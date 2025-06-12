#pragma once

#include <mutex>
#include <optional>
#include <sensor_msgs/JointState.h> //this is generic ROS message header, needed for gazebo sim
#include <std_msgs/Bool.h>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/robot_hw.h>
#include <hardware_interface/joint_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <ros/time.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Time.h>
#include <Eigen/Core>

// from my linermpc in drake c++
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

namespace linearmpc_panda {
    #define NUM_JOINTS 7
    #define PI 3.14

    class LinearMPCController: public controller_interface::MultiInterfaceController<
            franka_hw::FrankaModelInterface,
            franka_hw::FrankaStateInterface,
            hardware_interface::EffortJointInterface> 
{
    public:
        /* I should initialize my prog, ref_trajs, etc*/
        bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;

        //
        void update(const ros::Time& time, const ros::Duration& period) override;

        //
        void starting(const ros::Time& time) override;

        //
        void executor_callback(const std_msgs::Float64MultiArray::ConstPtr& msg);

        //
        bool initial_pose_ok(const Eigen::VectorXd& q_init_desired); 

        //
        //void q_init_callback(const sensor_msgs::JointState::ConstPtr& msg);

    private:
        ros::NodeHandle nh_;
        /* TODO: make the matrix or vector size explicit where possible! Some are dependent on 
                 MPC loop parameters, find a way to fix its size accordingly in the constructor */
        ros::Subscriber executor_sub_; // sub to set u_cmd at 1kHz
        //ros::Subscriber q_init_desired_sub_; // use this to check if the tracking is ready to be started
        ros::Publisher mpc_t_start_pub_;
        //ros::Publisher q_init_flag_pub_;        
        
        std::unique_ptr <franka_hw::FrankaModelHandle> model_handle_;
        std::unique_ptr <franka_hw::FrankaStateHandle> state_handle_;
        std::vector<hardware_interface::JointHandle> joint_handles_;
        Eigen::Matrix<double, NUM_JOINTS, 1> q_now_ {};
        Eigen::Matrix<double, NUM_JOINTS, 1> v_now_ {};
        Eigen::Matrix<double, NUM_JOINTS, 1> u_now_ {};

        franka::RobotState robot_state_ {};
        Eigen::VectorXd u_cmd_ {};
        std::mutex u_cmd_mutex_;
        // std_msgs::Time mpc_t_start_msg_ {};
        Eigen::VectorXd q_init_desired_ {};
        bool u_cmd_received_ {false};

    };
} // namespace linearmpc_panda

//xo =  0.770901  0.396021 -0.812618  -2.17939  0.663888   2.34041      -0.5