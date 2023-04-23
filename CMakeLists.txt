cmake_minimum_required(VERSION 3.0.2)
project(project)

find_package(catkin REQUIRED COMPONENTS
  nav_core
  pluginlib
  geometry_msgs
  std_msgs
  nav_msgs
  roscpp
  rospy
  tf2
  tf2_ros
  tf2_geometry_msgs
  base_local_planner
  message_generation
  
)

find_package(Eigen3 REQUIRED)



## Generate messages in the 'msg' folder
#add_message_files(
#  FILES
#  my_msg.msg
#)

## Generate services in the 'srv' folder
add_service_files(
    FILES
    roi.srv
)

generate_messages(DEPENDENCIES std_msgs )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES project
  CATKIN_DEPENDS nav_core rospy  pluginlib geometry_msgs std_msgs nav_msgs roscpp tf2 tf2_ros tf2_geometry_msgs base_local_planner message_runtime
#  DEPENDS system_lib
  DEPENDS EIGEN3
)

#catkin_python_setup()

catkin_install_python(PROGRAMS scripts/send_roi.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
	include 
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIRS}
)

add_library(rrt_global_planner_lib src/rrt_global_planner.cpp src/TreeNode.cpp)
target_link_libraries(rrt_global_planner_lib ${catkin_LIBRARIES})


add_library(rrt_star_global_planner_lib src/random_double_generator.cpp src/rrt_star_global_planner.cpp src/rrt_star.cpp src/collision_detector.cpp)
target_link_libraries(rrt_star_global_planner_lib ${catkin_LIBRARIES})


add_library(llc_local_planner_lib src/llc_local_planner.cpp)
target_link_libraries(llc_local_planner_lib ${catkin_LIBRARIES})

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
