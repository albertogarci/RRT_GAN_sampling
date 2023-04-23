#ifndef LLC_LOCAL_PLANNER_H
#define LLC_LOCAL_PLANNER_H

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf/tf.h>

#include <angles/angles.h>



namespace llc_local_planner{

class LLCLocalPlanner: public nav_core::BaseLocalPlanner{

	costmap_2d::Costmap2DROS* costmap_ros_;
	costmap_2d::Costmap2D* costmap_;
	tf2_ros::Buffer* tf_;	//for tf transformations

	std::vector<geometry_msgs::PoseStamped> global_plan_;

	double kalpha_, krho_, kbeta_;
	double rho_th_;

	double robot_radius_;

	bool initialized_;

public:

	LLCLocalPlanner() : costmap_(NULL), initialized_(false){};
	LLCLocalPlanner(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);

	/*override functions from interface nav_core::BaseLocalPlanner*/
	void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);
	bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);
	bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
	bool isGoalReached();

private:

};

};

#endif
