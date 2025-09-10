#pragma once
#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <realtime_tools/realtime_buffer.h>
#include <Eigen/Dense>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>

//#include <franka_example_controllers/JointTorqueComparison.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/trigger_rate.h>

namespace franka_torque_controller {
 
#define NUM_JOINTS 7
#define PI 3.14
using Vec7 = Eigen::Matrix<double,7,1>;

class TorquePDController : public controller_interface::MultiInterfaceController<
                                    franka_hw::FrankaModelInterface,
                                    franka_hw::FrankaStateInterface,
                                    hardware_interface::EffortJointInterface> 
{
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;

  void update(const ros::Time& time, const ros::Duration& period) override;

  void starting(const ros::Time& time) override;

  //void stopping(const ros::Time& time) override;

  // some helper functions
  //#################################################################################
  Eigen::VectorXd cumulative_sum(const Eigen::VectorXd& vec) 
  {
    Eigen::VectorXd result(vec.size());
    double acc = 0.0;
    for (int i = 0; i < vec.size(); ++i) {
        acc += vec(i);
        result(i) = acc;
    }
    return result;
  }

  //#################################################################################
  // Load from CSV
  Eigen::MatrixXd load_csv(const std::string& filename, int rows, int cols) 
  {
    std::ifstream in(filename);
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        std::string line;
        std::getline(in, line);
        std::stringstream ss(line);
        std::string cell;
        for (int j = 0; j < cols; ++j) {
            std::getline(ss, cell, ',');
            mat(i,j) = std::stod(cell);
        }
    }
    return mat;
  }

 //#################################################################################
  template <typename T>
  class LinearSpline {
  public:
  /*  // constructor, must construct with ts and ys 
      LinearSpline(std::vector<double> ts, std::vector<T> ys)
          : knots_(std::move(ts)), values_(std::move(ys))
      {
          if (knots_.size() != values_.size() || knots_.size() < 2)
              throw std::invalid_argument("Need same number of times/values >= 2");
          if (!std::is_sorted(knots_.begin(), knots_.end()))
              throw std::invalid_argument("Times must be strictly increasing");
      }
  */
      LinearSpline() = default; // allow default construction

      LinearSpline(std::vector<double> ts, std::vector<T> ys) {
          reset(std::move(ts), std::move(ys));
      }

      void reset(std::vector<double> ts, std::vector<T> ys) {
          knots_  = std::move(ts);
          values_ = std::move(ys);
          if (knots_.size() != values_.size() || knots_.size() < 2)
              throw std::invalid_argument("Need same number of times/values >= 2");
          if (!std::is_sorted(knots_.begin(), knots_.end()))
              throw std::invalid_argument("Times must be strictly increasing");
          last_idx_ = 0;
      }

      // Evaluate at time t
      T operator()(double t) const {
          if (t <= knots_.front()) return values_.front();
          if (t >= knots_.back())  return values_.back();

          // Fast path: increment cached index if query moved forward
          while (last_idx_ + 1 < knots_.size() && t > knots_[last_idx_+1]) {
              ++last_idx_;
          }
          // If query moved backwards, reset cache and search again
          while (last_idx_ > 0 && t < knots_[last_idx_]) {
              --last_idx_;
          }

          double x0 = knots_[last_idx_];
          double x1 = knots_[last_idx_+1];
          double alpha = (t - x0) / (x1 - x0);
          return (1.0 - alpha) * values_[last_idx_] + alpha * values_[last_idx_+1];
      }

      // Piecewise-constant derivative
      T derivative(double t) const {
          if (t <= knots_.front()) return slope(0);
          if (t >= knots_.back())  return slope(knots_.size()-2);

          // Reuse the cached index logic
          operator()(t); // updates last_idx_ for this t
          return slope(last_idx_);
    }

  private:
      T slope(std::size_t i) const {
          double dt = knots_[i+1] - knots_[i];
          return (values_[i+1] - values_[i]) / dt;
      }

      std::vector<double> knots_;
      std::vector<T> values_;
      mutable std::size_t last_idx_ = 0; // cache, mutable for const operator()
  }; 

 private:
  Eigen::Matrix<double, 7, 1> SaturateTorqueRate(
                                    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
                                    const Eigen::Matrix<double, 7, 1>& tau_J_d);

  void u_cmd_callback(const std_msgs::Float64MultiArray::ConstPtr& msg);

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  std::vector<std::string> joint_names_ {
      "panda_joint1", "panda_joint2", "panda_joint3", 
      "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
  } ;

  franka_hw::TriggerRate trigger_rate_ {0.1}; // 10 Hz
  static constexpr double kMaxTorqueRate = 0.1; 
  const double delta_tau_max_ {1.0}; 
  std::array<double, 7> last_tau_d_ {};
  std::array<double, 7> dq_filtered_ {};

  // sub and pub 
  ros::Subscriber u_cmd_subscriber_;
  realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray> torque_publisher_;

  // path for ref trajs
  // task1
  //std::string ref_traj_path_h_ {"/home/sc19zx/catkin_ws/experiments/task1_N60_euler_hlow0.07_dtheta2.0/test1_N60_Euler_hlow0.07_h.csv"};
  //std::string ref_traj_path_q_ {"/home/sc19zx/catkin_ws/experiments/task1_N60_euler_hlow0.07_dtheta2.0/test1_N60_Euler_hlow0.07_q.csv"};
  //std::string ref_traj_path_v_ {"/home/sc19zx/catkin_ws/experiments/task1_N60_euler_hlow0.07_dtheta2.0/test1_N60_Euler_hlow0.07_v.csv"};
  //std::string ref_traj_path_u_ {"/home/sc19zx/catkin_ws/experiments/task1_N60_euler_hlow0.07_dtheta2.0/test1_N60_Euler_hlow0.07_u.csv"};
 
  // task2
  //std::string ref_traj_path_h_ {"/home/sc19zx/catkin_ws/experiments/task2_N60_euler_hlow0.04_dtheta2.0/test2_N60_Euler_hlow0.04_h.csv"};
  //std::string ref_traj_path_q_ {"/home/sc19zx/catkin_ws/experiments/task2_N60_euler_hlow0.04_dtheta2.0/test2_N60_Euler_hlow0.04_q.csv"};
  //std::string ref_traj_path_v_ {"/home/sc19zx/catkin_ws/experiments/task2_N60_euler_hlow0.04_dtheta2.0/test2_N60_Euler_hlow0.04_v.csv"};
  //std::string ref_traj_path_u_ {"/home/sc19zx/catkin_ws/experiments/task2_N60_euler_hlow0.04_dtheta2.0/test2_N60_Euler_hlow0.04_u.csv"};
 
  // task3
  std::string ref_traj_path_h_ {"/home/sc19zx/catkin_ws/experiments/task3_N60_euler_hlow0.03_dtheta1.0/test3_N60_Euler_hlow0.03_h.csv"};
  std::string ref_traj_path_q_ {"/home/sc19zx/catkin_ws/experiments/task3_N60_euler_hlow0.03_dtheta1.0/test3_N60_Euler_hlow0.03_q.csv"};
  std::string ref_traj_path_v_ {"/home/sc19zx/catkin_ws/experiments/task3_N60_euler_hlow0.03_dtheta1.0/test3_N60_Euler_hlow0.03_v.csv"};
  std::string ref_traj_path_u_ {"/home/sc19zx/catkin_ws/experiments/task3_N60_euler_hlow0.03_dtheta1.0/test3_N60_Euler_hlow0.03_u.csv"};

  // initialize the splines for q, v, u to be defined in starting() later
  LinearSpline<Vec7> q_spline_;
  LinearSpline<Vec7> v_spline_;
  LinearSpline<Vec7> u_spline_;

  // starting time 
  double t_traj_;
  Eigen::VectorXd Kp_ = Eigen::VectorXd::Constant(NUM_JOINTS, 15.);
  Eigen::VectorXd Kd_ = Eigen::VectorXd::Constant(NUM_JOINTS, 5.);
  double alpha_ = 0.99;

  // add a publisher for the starting time
  ros::Publisher start_time_publisher_;
  ros::Publisher traj_completion_pub_;
  double traj_completion_time_ = 0.0;
  bool traj_complete_published_ = false;
  double t_delay_ = 0.1; // 100ms delay to ensure trajectory completion
};
}


