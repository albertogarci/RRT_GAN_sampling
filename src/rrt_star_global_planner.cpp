#include <pluginlib/class_list_macros.h>
#include <time.h>
#include <chrono>

#include "AROB_lab4/rrt_star_global_planner/rrt_star_global_planner.hpp"

// register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(rrt_star_global_planner::RRTStarPlanner, nav_core::BaseGlobalPlanner)

namespace rrt_star_global_planner 
{

  RRTStarPlanner::RRTStarPlanner() : costmap_(nullptr), initialized_(false) {}

  RRTStarPlanner::RRTStarPlanner(std::string name,
                                costmap_2d::Costmap2DROS* costmap_ros) : costmap_(nullptr), initialized_(false) {
    // initialize the planner
    initialize(name, costmap_ros);
  }

  RRTStarPlanner::RRTStarPlanner(std::string name,
                                costmap_2d::Costmap2D* costmap,
                                std::string global_frame) : costmap_(nullptr), initialized_(false) {
    // initialize the planner
    initialize(name, costmap, global_frame);
  }

  void RRTStarPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
    initialize(name, costmap_ros->getCostmap(), costmap_ros->getGlobalFrameID());
  }

  void RRTStarPlanner::initialize(std::string name, costmap_2d::Costmap2D* costmap, std::string global_frame) {
    if (!initialized_) {
      costmap_ = costmap;
      global_frame_ = global_frame;

      ros::NodeHandle private_nh("~/" + name);
      private_nh.param("goal_tolerance", goal_tolerance_, 0.5);
      private_nh.param("radius", radius_, 2.0);
      private_nh.param("epsilon", epsilon_, 0.2);
      private_nh.param("max_num_nodes", max_num_nodes_, 5000);
      private_nh.param("min_num_nodes", min_num_nodes_, 500);
			std::cout << "Initialize ..." << std::endl;
      
        map_width_ = costmap_->getSizeInMetersX();
        map_height_ = costmap_->getSizeInMetersY();
      
      std::cout << "map_width: " << map_width_ << ", map_height: " << map_height_ << std::endl;

      ros::NodeHandle node_handle("~/" + name);
      vis_pub = node_handle.advertise<visualization_msgs::Marker>(node_handle.resolveName("/rrt_marker"), 1000);


    ros::NodeHandle n;
    client = n.serviceClient<project::roi>("ROI");


      ROS_INFO("RRT* Global Planner initialized successfully.");
      initialized_ = true;
    } else {
      ROS_WARN("This planner has already been initialized... doing nothing.");
    }
  }

  bool RRTStarPlanner::makePlan(const geometry_msgs::PoseStamped& start,
                                const geometry_msgs::PoseStamped& goal,
                                std::vector<geometry_msgs::PoseStamped>& plan) {
    // clear the plan, just in case
    plan.clear();

    ROS_INFO("RRT* Global Planner");
    ROS_INFO("Current Position: ( %.2lf, %.2lf)", start.pose.position.x, start.pose.position.y);
    ROS_INFO("GOAL Position: ( %.2lf, %.2lf)", goal.pose.position.x, goal.pose.position.y);

    std::pair<float, float> start_point = {start.pose.position.x, start.pose.position.y};
    std::pair<float, float> goal_point = {goal.pose.position.x, goal.pose.position.y};


    project::roi srv;
    srv.request.start_x =  start.pose.position.x;
    srv.request.start_y =  start.pose.position.y;
    srv.request.goal_x =  goal.pose.position.x;
    srv.request.goal_y =  goal.pose.position.y;

    std::vector<std::pair<float, float>> roi;

    if (client.call(srv))
    {
      //ROS_INFO("Response:");
      std::vector<unsigned int> res = srv.response.data;
      for (int i = 0; i < res.size(); i+=2) {
        std::pair<float, float> pair;
 
        pair.first = (static_cast< float >(res.at(i+1)) / 4.);
        pair.second = (static_cast< float >(res.at(i)) / 4.) ;

        if (pair.first > 8) {
          pair.first -= 8;
        } else if (pair.first < 8){
          pair.first = - (8 - pair.first);
        } else{
          pair.first = 0;

        }
        if (pair.second < 8) {
          pair.second = (8 - pair.second);

        } else if (pair.second > 8){
          pair.second = - ( pair.second - 8);


        } else{
          pair.second = 0;

        }
          //std::cout <<  pair.first << ", " << pair.second << "; " << std::flush;
          roi.push_back(pair);

      }
    }
    else
    {
      ROS_ERROR("Failed to call service add_two_ints");
      return 1;
    }
    //std::vector<std::pair<float, float>> roi_empty;
    //std::vector<unsigned int> roiv;
    planner_ = std::shared_ptr<RRTStar>(new RRTStar(start_point,
                                                    goal_point,
                                                    costmap_,
                                                    goal_tolerance_,
                                                    radius_,
                                                    epsilon_,
                                                    max_num_nodes_,
                                                    min_num_nodes_,
                                                    map_width_,
                                                    map_height_,
                                                    roi));

    std::list<std::pair<float, float>> path;

  auto start_t = std::chrono::steady_clock::now();
    if (planner_->pathPlanning(path)) {
      ROS_INFO("RRT* Global Planner: Path found!!!!");
      computeFinalPlan(plan, path);
      auto end = std::chrono::steady_clock::now();
      auto diff = end - start_t;
      std::cout << "-> Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / 1000. << " ms" << std::endl;

      return true;
    } else {
      ROS_WARN("The planner failed to find a path, choose other goal position");
      return false;
    }



  }

  void  RRTStarPlanner::computeFinalPlan(std::vector<geometry_msgs::PoseStamped>& plan,
                                        const std::list<std::pair<float, float>> &path) {
    // clean plan
    plan.clear();
    ros::Time plan_time = ros::Time::now();
    bool first = true;
    double total_dist = 0;
    std::pair<double, double> first_p, second_p;

    // convert points to poses
    for (const auto &point : path) {
      geometry_msgs::PoseStamped pose;
      pose.header.stamp = plan_time;
      pose.header.frame_id = global_frame_;
      pose.pose.position.x = point.first;
      pose.pose.position.y = point.second;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.x = 0.0;
      pose.pose.orientation.y = 0.0;
      pose.pose.orientation.z = 0.0;
      pose.pose.orientation.w = 1.0;
      plan.push_back(pose);
      if (first){
        first = false;
        first_p = point;
      } else{
        second_p = point;
        double d = euclideanDistance2D(first_p.first,
                              first_p.second,
                              second_p.first,
                              second_p.second);
        first_p = second_p;
        total_dist += d;
      }
    }

    std::cout << "-----> Path found, total_dist:" << total_dist << std::endl;

    //Visualize solution
    int i = 0;
    ros::Rate r(0.01);
    visualization_msgs::Marker markerD;
        markerD.action = visualization_msgs::Marker::DELETEALL;
        markerD.header.frame_id = global_frame_;
        vis_pub.publish( markerD );

    for(geometry_msgs::PoseStamped pos : plan) {
        visualization_msgs::Marker marker;

        marker.header.frame_id = global_frame_;
        marker.header.stamp = ros::Time();
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.ns = "/marker";
        marker.id = i;

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        //std::cout << "--> Marker! "<< i  << ", " << pos.pose.position.x << ", " <<  pos.pose.position.y << std::endl;

        marker.pose.position.x = pos.pose.position.x;
        marker.pose.position.y = pos.pose.position.y;
        marker.pose.position.z = pos.pose.position.z;

        marker.scale.x = 0.1;
        marker.scale.z = 0.1;
        marker.scale.y = 0.1;

        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration();


        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        vis_pub.publish( marker );
        i++;
    }

  }

}  
