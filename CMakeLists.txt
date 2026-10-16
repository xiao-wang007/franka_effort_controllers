cmake_minimum_required(VERSION 3.4)
project(panda_torque_controllers)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  std_msgs
  dynamic_reconfigure
  eigen_conversions
  franka_hw
  franka_gripper
  geometry_msgs
  hardware_interface
  joint_limits_interface
  tf
  tf_conversions
  message_generation
  pluginlib
  realtime_tools
  roscpp
  rospy
  trajectory_msgs
  urdf
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()


#generate_dynamic_reconfigure_options(
#  cfg/compliance_param.cfg
#  cfg/desired_mass_param.cfg
#  cfg/dual_arm_compliance_param.cfg
#  cfg/teleop_param.cfg
#  cfg/teleop_gripper_param.cfg
#)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES panda_torque_controllers
  CATKIN_DEPENDS
    controller_interface
  std_msgs
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    geometry_msgs
    hardware_interface
    joint_limits_interface
    tf
    tf_conversions
    message_runtime
    pluginlib
    realtime_tools
    roscpp
    trajectory_msgs
    urdf
    visualization_msgs
  DEPENDS Franka
)

add_library(panda_torque_controllers
  src/panda_mpc_torque_controller.cc
  src/panda_torque_pd_controller.cc
  src/panda_torque_pd_controller_simpson.cc
  src/panda_cartesian_trajectory_controller.cc
)

add_dependencies(panda_torque_controllers
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  # ${PROJECT_NAME}_gencfg
)

target_link_libraries(panda_torque_controllers PUBLIC
  ${Franka_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${catkin_LIBRARIES}
)

target_include_directories(panda_torque_controllers SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(panda_torque_controllers PUBLIC
  include
)


## Installation
install(TARGETS panda_torque_controllers
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES panda_torque_controller_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# #this is same as above
# pluginlib_export_plugin_description_file(controller_interface panda_torque_controller_plugins.xml)

# catkin_install_python(
#   PROGRAMS scripts/interactive_marker.py scripts/move_to_start.py scripts/dual_arm_interactive_marker.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(panda_torque_controllers FILES ${SOURCES} ${HEADERS})
  add_tidy_target(panda_torque_controllers
    FILES ${SOURCES}
    DEPENDS panda_torque_controllers
  )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/PepTools.cmake OPTIONAL
  RESULT_VARIABLE PEP_TOOLS
)
if(PEP_TOOLS)
  file(GLOB_RECURSE PYSOURCES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)
  add_pyformat_target(franka_control FILES ${PYSOURCES})
endif()

# Bundle the Eigen-only core so the robot workstation needs no Drake checkout.
# The existing PUBLIC include/ directory resolves <cartesian/core.h>.
install(DIRECTORY include/cartesian DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})

# Recorder test does not require a robot or a ROS master.
if(CATKIN_ENABLE_TESTING)
  add_executable(simpson_recording_test tests/simpson_recording_test.cc)
  target_include_directories(simpson_recording_test PRIVATE include)
  target_compile_options(simpson_recording_test PRIVATE -UNDEBUG)
  target_link_libraries(simpson_recording_test ${CMAKE_THREAD_LIBS_INIT})
  add_test(NAME simpson_recording_test COMMAND simpson_recording_test)
  add_executable(cartesian_core_test tests/cartesian_core_test.cc)
  target_include_directories(cartesian_core_test PRIVATE include)
  target_include_directories(cartesian_core_test SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIRS})
  target_compile_options(cartesian_core_test PRIVATE -UNDEBUG)
  add_test(NAME cartesian_core_test COMMAND cartesian_core_test)
endif()
# Existing include directory is panda_mpc, rather than the package name.
install(DIRECTORY include/panda_mpc DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
