// #include "panda_QP_controller.h"
#include <linearmpc_panda/panda_mpc_torque_controller.h>
#include <pluginlib/class_list_macros.h>
#include <franka/robot_state.h>
#include <franka_hw/franka_model_interface.h>
#include <hardware_interface/hardware_interface.h>

namespace linearmpc_panda {

	//#######################################################################################
    bool LinearMPCController::init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &node_handle) 
	{
		nh_ = node_handle;

        //Get Franka model and state interfaces
        auto *model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
        auto *state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
        auto *effort_interface = robot_hw->get<hardware_interface::EffortJointInterface>();

        if (!model_interface || !state_interface || !effort_interface) {
            ROS_ERROR("Failed to get required hardware interfaces.");
            return false;
        }

        // Get handles
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle("panda_model"));
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle("panda_robot"));

        auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();

		joint_handles_.clear();
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint1"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint2"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint3"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint4"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint5"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint6"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint7"));

        //// Convert current robot position and velocity into Eigen data storage
        //Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> q_now_(robot_state_.q.data());
        //Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> v_now_(robot_state_.dq.data());
		//Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> u_now_(robot_state_.tau_J.data());

		q_init_desired_sub_ = node_handle.subscribe("/q_init_desired", 1, &LinearMPCController::q_init_callback, this) // should make sure the publisher is latched

		//get upsampled solution trajectory, at hardware frequency 1kHz 
		executor_sub_ = node_handle.subscribe("/upsampled_u_cmd", 1, &LinearMPCController::executor_callback, this);
		// mpc_t_start_pub_ = node_handle.advertise<std_msgs::Time>("/mpc_t_start", 1, true); //True for latched publisher
		q_init_flag_pub_ = node_handle.advertise<std_msgs::Bool>("/q_init_ok", 1, true); //True for latched publisher

		u_cmd_ = Eigen::VectorXd::Zero(NUM_JOINTS);

		ROS_INFO("\n Linear MPC Controller initialized successfully. xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n");
        return true;
    }

	//#######################################################################################
    void LinearMPCController::starting(const ros::Time& time) 
    {
		// Wait for the first message (timeout after 1.0 sec)
		auto msg = ros::topic::waitForMessage<sensor_msgs::JointState>("/franka_state_controller/joint_states", nh_, ros::Duration(1.0));
		Eigen::Map<const Eigen::VectorXd> qs(msg->position.data(), msg->position.size());
		if (msg) {
			ROS_INFO("First message received: %f", msg->data);
		} else {
			ROS_WARN("No message received within timeout!");
		}

        // set the condition here to check if current robot state is close to x0_ref
		if (!initial_pose_ok(qs))
		{
			std_msgs::Bool q_init_ok_msg;
			q_init_ok_msg.data = false;
			q_init_flag_pub_.publish(q_init_ok_msg);
			ROS_WARN("Initial pose is not ok, set the flag to call service");

			//block untial the robot reaches q_init_desired_
			ros::Rate rate(100); // 100 Hz
			while (ros::ok())
			{
				auto msg = ros::topic::waitForMessage<sensor_msgs::JointState>("/franka_state_controller/joint_states", nh_, ros::Duration(1.0));
				if (!msg)
				{
					ROS_WARN("Waiting for joint states msg...");
					rate.sleep();
					continue;
				}
				Eigen::Map<const Eigen::VectorXd> qs(msg->position.data(), msg->position.size());
				if (initial_pose_ok(qs)) 
				{
					ROS_INFO("Initial pose is reached! Setting the flag...");
					q_init_ok_msg.data = true;
					q_init_flag_pub_.publish(q_init_ok_msg);
					break; // exit the loop when the initial pose is ok
				}
				else 
				{
					ROS_WARN("Waiting for initial pose to be reached...");
					rate.sleep();
				}
			}
		}

    }

	//#######################################################################################
    void LinearMPCController::update(const ros::Time& time, const ros::Duration& period) 
	{
		if (!u_cmd_received_) 
		{
			ROS_WARN("No u_cmd received yet, skipping update.");
			return; // Skip update if no command has been received
		}

		Eigen::VectorXd u_cmd_copy;
		{
			std::lock_guard<std::mutex> lock(u_cmd_mutex_);
			u_cmd_copy = u_cmd_;
		}

		for (size_t i = 0; i < NUM_JOINTS; i++) 
		{
			joint_handles_[i].setCommand(u_cmd_copy[i]); //u_cmd_ is from the sub
		}

    }

	//#######################################################################################
	bool LinearMPCController::initial_pose_ok(const Eigen::VectorXd& q_init_desired) 
	{
		// Check if the current robot state is close to the desired initial pose
		const double threshold = 0.1; // Adjust this threshold as needed

		if ((q_now_ - q_init_desired).norm() > threshold) 
		{
			ROS_WARN("Current robot state is not close to the desired initial pose.");
			return false;
		}
		else if ((q_now_ - q_init_desired).norm() < threshold)
		{
			ROS_INFO("Current robot state is close to the desired initial pose.");
			return true;
		}
	}

	//#######################################################################################
	void LinearMPCController::executor_callback(const std_msgs::Float64MultiArray::ConstPtr& dim7_vec_msg) 
	{
		
		// ROS_INFO("checking inside executor_callback(), 1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n");
		//get the upsampled solution
		if (!u_cmd_received_)
		{
			return;
		}
		else
		{
			std::lock_guard<std::mutex> lock(u_cmd_mutex_);
			u_cmd_ = Eigen::Map<const Eigen::VectorXd>(dim7_vec_msg->data.data(), dim7_vec_msg->data.size());
			u_cmd_received_ = true;
		}
	}

	//#######################################################################################
	void LinearMPCController::q_init_callback(const std_msgs::Float64MultiArray::ConstPtr& dim7_vec_msg) 
	{
		
		// ROS_INFO("checking inside executor_callback(), 1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n");
		//get the upsampled solution
		if (dim7_vec_msg->data.size() == 0)
		{
			return;
		}
		else
		{
			q_init_desired_ = Eigen::Map<const Eigen::VectorXd>(dim7_vec_msg->data.data(), dim7_vec_msg->data.size());
		}
	}

} //namespace linearmpc_panda

PLUGINLIB_EXPORT_CLASS(linearmpc_panda::LinearMPCController, controller_interface::ControllerBase)

