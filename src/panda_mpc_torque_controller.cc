// #include "panda_QP_controller.h"
#include <panda_mpc/panda_mpc_torque_controller.h>
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
		// initialize model and state handles
		try 
		{
        	model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle("panda_model"));
		} catch (const hardware_interface::HardwareInterfaceException& e) 
		{
			ROS_ERROR_STREAM("Failed to get Franka model handle: " << e.what());
			return false;
		}

		try 
		{
			state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle("panda_robot"));
		} 
		catch (const hardware_interface::HardwareInterfaceException& e) 
		{
			ROS_ERROR_STREAM("Failed to get Franka state handle: " << e.what());
			return false;

		}

        auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();

		joint_handles_.clear();
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint1"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint2"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint3"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint4"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint5"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint6"));
        joint_handles_.push_back(effort_joint_interface->getHandle("panda_joint7"));


		//get upsampled solution trajectory, at hardware frequency 1kHz 
		executor_sub_ = node_handle.subscribe("/upsampled_u_cmd", 1, &LinearMPCController::executor_callback, this);
		tau_J_d_pub_ = node_handle.advertise<std_msgs::Float64MultiArray>("/tau_J_d", 1);
		q_init_desired_sub_ = node_handle.subscribe("/q_init_desired", 1, &LinearMPCController::q_init_desired_callback, this);

		u_cmd_ = Eigen::VectorXd::Zero(NUM_JOINTS);

		ROS_INFO("\n Linear MPC Controller initialized successfully. xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n");
        return true;
    }

	//#######################################################################################
    void LinearMPCController::starting(const ros::Time& time) 
    {
		running_ = true;
		tau_J_d_pub_thread_ = std::thread([this]() 
		{
			publish_tau_J_d_loop();
		});
		tau_J_d_pub_thread_.detach();

		robot_state_ = state_handle_->getRobotState();
		
        // Convert current robot position and velocity into Eigen data storage
        Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> q_now_(robot_state_.q.data());
        Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> v_now_(robot_state_.dq.data());
		Eigen::Map<Eigen::Matrix<double, NUM_JOINTS, 1>> u_now_(robot_state_.tau_J.data());

		// print current robot state just to notify
		ROS_INFO_STREAM("Current robot state: \n"
			<< "q_now: " << q_now_.transpose() << "\n"
			<< "v_now: " << v_now_.transpose() << "\n"
			<< "u_now: " << u_now_.transpose() << "\n");
    }

	//#######################################################################################
	void LinearMPCController::publish_tau_J_d_loop() 
	{ 
		std_msgs::Float64MultiArray msg;
		while (running_) 
		{
			std::array<double, 7> tau;
			while (tau_J_d_queue_.pop(tau)) 
			{
				msg.data.assign(tau.begin(), tau.end());
				tau_J_d_pub_.publish(msg);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));  // or use ros::Rate
		}
	}


	//#######################################################################################
    void LinearMPCController::update(const ros::Time& time, const ros::Duration& period) 
	{
		franka::RobotState robot_state = state_handle_->getRobotState();
		Eigen::VectorXd q_now = Eigen::Map<const Eigen::VectorXd>(robot_state.q.data(), NUM_JOINTS);
		Eigen::VectorXd v_now = Eigen::Map<const Eigen::VectorXd>(robot_state.dq.data(), NUM_JOINTS);

		if (!u_cmd_received_) 
		{
			ROS_WARN_THROTTLE(1.0, "No u_cmd received yet, sending zero torques to joints.");
			//auto g_comp = model_handle_->getGravity();


			/* if use auto here, the type is CwiseBinaryOp. But with explicity type of Eigen::VectorXd
			   Eigen to evaluate the expression immediately and store the result as a concrete 
			   Eigen::VectorXd object. This triggers Eigen’s implicit conversion operator, which 
			   takes the lazily evaluated CwiseBinaryOp (the expression template) and performs 
			   the actual computation into memory. */
			//auto tau_stabilization = kp_.array() * (q_init_desired_ - q_now).array() 
								     //+ kd_.array() * (v_init_desired_ - v_now).array();
			Eigen::VectorXd tau_stabilization = kp_.array() * (q_init_desired_ - q_now).array() 
								                + kd_.array() * (v_init_desired_ - v_now).array();
			this->saturateTorqueRate(tau_stabilization, robot_state.tau_J_d);

			for (size_t i = 0; i < NUM_JOINTS; i++)
			{
				//joint_handles_[i].setCommand(g_comp[i]); // Set gravity compensation torque
				//joint_handles_[i].setCommand(0.0); // Set zero torque if no command received
				joint_handles_[i].setCommand(tau_stabilization(i)); // Set zero torque if no command received
			}
		}

		if (!tau_J_d_queue_.push(robot_state.tau_J_d)) 
		{
			ROS_WARN_THROTTLE(1.0, "tau_J_d_queue_ is full, dropping the oldest command.");
		}

		Eigen::VectorXd u_cmd_copy;
		{
			std::lock_guard<std::mutex> lock(u_cmd_mutex_);
			u_cmd_copy = u_cmd_;
		}

		// avoid saturating the torque rate
		this->saturateTorqueRate(u_cmd_copy, robot_state.tau_J_d);

		for (size_t i = 0; i < NUM_JOINTS; i++) 
		{
			joint_handles_[i].setCommand(u_cmd_copy(i)); //u_cmd_ is from the sub
		}

		// reset the flag after sending the command
		//u_cmd_received_ = false;

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

		ROS_INFO("Current robot state is exactly at the threshold.");
		return true;  // or false, depending on your logic
	}

	//#######################################################################################
	void LinearMPCController::executor_callback(const std_msgs::Float64MultiArray::ConstPtr& dim7_vec_msg) 
	{
		
		// ROS_INFO("checking inside executor_callback(), 1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n");
		//get the upsampled solution
		std::lock_guard<std::mutex> lock(u_cmd_mutex_);
		u_cmd_ = Eigen::Map<const Eigen::VectorXd>(dim7_vec_msg->data.data(), dim7_vec_msg->data.size());
		u_cmd_received_ = true;
	}

	//#######################################################################################
    void LinearMPCController::q_init_desired_callback(const sensor_msgs::JointState::ConstPtr& msg)
	{
		q_init_desired_.resize(NUM_JOINTS);
		v_init_desired_.resize(NUM_JOINTS);
		q_init_desired_ = Eigen::Map<const Eigen::VectorXd>(msg->position.data(), msg->position.size());
		v_init_desired_ = Eigen::Map<const Eigen::VectorXd>(msg->velocity.data(), msg->velocity.size());
		
		ROS_INFO_STREAM("Received q_init_desired: " << q_init_desired_.transpose() << "\n"
		<< "Received v_init_desired: " << v_init_desired_.transpose() << "\n");
	}

	//#######################################################################################
	void LinearMPCController::saturateTorqueRate(Eigen::VectorXd& tau_cmd, const std::array<double, 7>& tau_measured)
	{
		for (size_t i = 0; i < NUM_JOINTS; i++)
		{
			double diff = tau_cmd(i) - tau_measured[i];
			tau_cmd(i) = tau_measured[i] + std::max(std::min(diff, dtau_up_), -dtau_up_);
		}
	}


	//#######################################################################################
	void LinearMPCController::stopping()
	{
		running_ = false;
		if (tau_J_d_pub_thread_.joinable()) 
		{
			tau_J_d_pub_thread_.join();
		}
	}

} //namespace linearmpc_panda

PLUGINLIB_EXPORT_CLASS(linearmpc_panda::LinearMPCController, controller_interface::ControllerBase)

