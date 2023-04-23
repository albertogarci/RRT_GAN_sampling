#ifndef RRT_STAR_GLOBAL_PLANNER_RRT_STAR_PLANNER_HPP_ 
#define RRT_STAR_GLOBAL_PLANNER_RRT_STAR_PLANNER_HPP_

#include <ros/ros.h>

#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>

#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <utility>
#include <memory>

#include "AROB_lab4/rrt_star_global_planner/node.hpp"
#include "AROB_lab4/rrt_star_global_planner/rrt_star.hpp"
#include "AROB_lab4/rrt_star_global_planner/random_double_generator.hpp"
#include "project/roi.h"

#include <visualization_msgs/Marker.h>

namespace rrt_star_global_planner {


class RRTStarPlanner : public nav_core::BaseGlobalPlanner {
 public:
  RRTStarPlanner();


  RRTStarPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  RRTStarPlanner(std::string name, costmap_2d::Costmap2D* costmap, std::string global_frame);
  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  void initialize(std::string name, costmap_2d::Costmap2D* costmap, std::string global_frame);


  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan); 

  void computeFinalPlan(std::vector<geometry_msgs::PoseStamped>& plan, 
                        const std::list<std::pair<float, float>> &path);

 private:
  costmap_2d::Costmap2D* costmap_{nullptr};
  bool initialized_{false};
  int max_num_nodes_;
  int min_num_nodes_;
  double epsilon_;
  double max_samples_;

  double map_width_;
  double width;
  double height;
  double global_width;
  double global_height;
  double max_dist_;
  double resolution_;

  double map_height_;
  double radius_;
  double goal_tolerance_;
  bool search_specific_area_{true};
  std::string global_frame_;
  std::shared_ptr<RRTStar> planner_;
  ros::Publisher vis_pub;
  ros::ServiceClient client;

};

}  

#endif   
