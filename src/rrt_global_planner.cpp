#include <pluginlib/class_list_macros.h>
#include "AROB_lab4/rrt_global_planner.h"
#include <random>
#include <chrono>

//register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(rrt_planner::RRTPlanner, nav_core::BaseGlobalPlanner)

//Default Constructor
namespace rrt_planner
{
    double distance(const std::vector<int> start, 
                                            const std::vector<int> end){

        double dist = std::sqrt(( (start[0] - end[0]) * (start[0] - end[0]))
                         + (start[1] - end[1]) * (start[1] - end[1]));

        return dist;

    }

    double distance(const unsigned int x0, const unsigned int y0, const unsigned int x1, const unsigned int y1)
    {
        return std::sqrt((int)(x1 - x0) * (int)(x1 - x0) + (int)(y1 - y0) * (int)(y1 - y0));
    }

    RRTPlanner::RRTPlanner() : costmap_ros_(NULL), initialized_(false),
                               max_samples_(0.0) {}

    RRTPlanner::RRTPlanner(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
    {
        initialize(name, costmap_ros);
    }

    void RRTPlanner::initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
    {

        if (!initialized_)
        {

            ros::NodeHandle node_handle("~/" + name);

			// std::cout << "Initialize ..." << std::endl;
			vis_pub = node_handle.advertise<visualization_msgs::Marker>(node_handle.resolveName("/rrt_marker"), 1000);

            ros::NodeHandle nh("~/" + name);
            ros::NodeHandle nh_local("~/local_costmap/");
            ros::NodeHandle nh_global("~/global_costmap/");

            nh.param("maxsamples", max_samples_, 10.);

            nh_global.param("width", global_width, 50.0);
            nh_global.param("height", global_height, 50.0);


            std::random_device rd; // obtain a random number from hardware
            eng = std::mt19937(rd());
            // seed the generator
            distr = std::uniform_int_distribution<std::mt19937::result_type>(0, global_width);   // define the range
            distr2 = std::uniform_int_distribution<std::mt19937::result_type>(0, global_height); // define the range
            
            //to make sure one of the nodes in the plan lies in the local costmap
            nh_local.param("width", width, 3.0);
            nh_local.param("height", height, 3.0);
            max_dist_ = (std::min(width, height) * 1.0); //or any other distance within local costmap

            nh_global.param("resolution", resolution_, 0.05);
            std::cout << "Parameters: " << max_samples_  << ", " << max_dist_ << std::endl;

            // std::cout << "Parameters: " << max_samples_ << ", " << dist_th_ << ", " << visualize_markers_ << ", " << max_dist_ << std::endl;
            // std::cout << "Local costmap size: " << width << ", " << height << std::endl;
            // std::cout << "Global costmap resolution: " << resolution_ << std::endl;

            costmap_ros_ = costmap_ros;
            costmap_ = costmap_ros->getCostmap();
            global_frame_id_ = costmap_ros_->getGlobalFrameID();

            initialized_ = true;
        }
        else
        {
            ROS_WARN("This planner has already been initialized... doing nothing.");
        }
    }

    bool RRTPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                              std::vector<geometry_msgs::PoseStamped> &plan)
    {

        std::cout << "RRTPlanner::makePlan" << std::endl;

        if (!initialized_)
        {
            ROS_ERROR("The planner has not been initialized.");
            return false;
        }

        if (start.header.frame_id != costmap_ros_->getGlobalFrameID())
        {
            ROS_ERROR("The start pose must be in the %s frame, but it is in the %s frame.",
                      global_frame_id_.c_str(), start.header.frame_id.c_str());
            return false;
        }

        if (goal.header.frame_id != costmap_ros_->getGlobalFrameID())
        {
            ROS_ERROR("The goal pose must be in the %s frame, but it is in the %s frame.",
                      global_frame_id_.c_str(), goal.header.frame_id.c_str());
            return false;
        }

        plan.clear();
        costmap_ = costmap_ros_->getCostmap(); // Update information from costmap

        // Get start and goal poses in map coordinates
        unsigned int goal_mx, goal_my, start_mx, start_my;
        if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_mx, goal_my))
        {
            ROS_WARN("Goal position is out of map bounds.");
            return false;
        }
        costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_mx, start_my);

        std::vector<int> point_start{(int)start_mx, (int)start_my};
        std::vector<int> point_goal{(int)goal_mx, (int)goal_my};
        std::vector<std::vector<int>> solRRT;
        bool computed = computeRRT(point_start, point_goal, solRRT);
        if (computed)
        {
            getPlan(solRRT, plan);
            // add goal
            plan.push_back(goal);
        }
        else
        {
            ROS_WARN("No plan computed");
        }

        return computed;
    }

    std::vector<int> RRTPlanner::createPoseWithinRange(const std::vector<int> start, 
                                            const std::vector<int> end, double range){
        double x_step = end[0] - start[0];
        double y_step = end[1] - start[1];
        double mag = std::sqrt((x_step * x_step) + (y_step * y_step));

        if (mag < range){
            return end;
        }

        x_step /= mag;
        y_step /= mag;

        std::vector<int> new_pose{static_cast<int>(start[0] + x_step * range), static_cast<int>(start[1] + y_step * range)};
        return new_pose;

    }


    bool RRTPlanner::computeRRT(const std::vector<int> start, const std::vector<int> goal,
                                std::vector<std::vector<int>> &sol)
    {
        bool finished = false;

        //Initialize random number generator
        srand(time(NULL));
        // Initialize the tree with the starting point in map coordinates
        // Add parent node
        TreeNode *itr_node = new TreeNode(start);
        auto start_t = std::chrono::steady_clock::now();

        int max_iterations = this->max_samples_ * 1000;
        int i = 0;
        TreeNode *goal_node = new TreeNode(goal);
        while (!finished && i < max_iterations)
        {
            //std::mt19937 eng(rd());
            TreeNode *x_new = TreeNode::GenerateRandomNode(distr, distr2, eng);
            TreeNode* x_near = x_new->neast(itr_node);
            auto new_pose = createPoseWithinRange(x_near->getNode(), x_new->getNode(), this->max_dist_);
            TreeNode *new_node = new TreeNode(new_pose);

            bool free = obstacleFree(new_node->getNode()[0], new_node->getNode()[1],
                                     x_near->getNode()[0], x_near->getNode()[1]);

            if (free)
            {
                x_near->appendChild(new_node);

                bool reach_goal = obstacleFree(new_node->getNode()[0], new_node->getNode()[1],
                                        goal_node->getNode()[0], goal_node->getNode()[1]);
                if (reach_goal){

                    new_node->appendChild(goal_node);

                    finished = true;
                    int count;
                    itr_node->countNodesRec(itr_node, count);

                    std::cout << "Path found, #nodes:" << count  << std::endl;
                    sol =  goal_node->returnSolution();
                    std::cout << "--> Plan done2! "<< std::endl;

                }

            }
            i++;
        }

      auto end = std::chrono::steady_clock::now();
      auto diff = end - start_t;
      std::cout << "-> Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / 100. << " ms" << std::endl;

        return finished;
    }

    bool RRTPlanner::obstacleFree(const unsigned int x0, const unsigned int y0,
                                  const unsigned int x1, const unsigned int y1)
    {
        //Bresenham algorithm to check if the line between points (x0,y0) - (x1,y1) is free of collision

        int dx = x1 - x0;
        int dy = y1 - y0;

        int incr_x = (dx > 0) ? 1.0 : -1.0;
        int incr_y = (dy > 0) ? 1.0 : -1.0;

        unsigned int da, db, incr_x_2, incr_y_2;
        if (abs(dx) >= abs(dy))
        {
            da = abs(dx);
            db = abs(dy);
            incr_x_2 = incr_x;
            incr_y_2 = 0;
        }
        else
        {
            da = abs(dy);
            db = abs(dx);
            incr_x_2 = 0;
            incr_y_2 = incr_y;
        }

        int p = 2 * db - da;
        unsigned int a = x0;
        unsigned int b = y0;
        unsigned int end = da;
        for (unsigned int i = 0; i < end; i++)
        {
            if (costmap_->getCost(a, b) != costmap_2d::FREE_SPACE)
            { // to include cells with inflated cost
                return false;
            }
            else
            {
                if (p >= 0)
                {
                    a += incr_x;
                    b += incr_y;
                    p -= 2 * da;
                }
                else
                {
                    a += incr_x_2;
                    b += incr_y_2;
                }
                p += 2 * db;
            }
        }

        return true;
    }

    void RRTPlanner::getPlan(const std::vector<std::vector<int>> sol, std::vector<geometry_msgs::PoseStamped> &plan)
    {
        bool first = true;
        double total_dist = 0;
        std::vector<int> first_p, second_p;

        for (auto it = sol.rbegin(); it != sol.rend(); it++)
        {
            std::vector<int> point = (*it);
            geometry_msgs::PoseStamped pose;

            costmap_->mapToWorld((unsigned int)point[0], (unsigned int)point[1],
                                 pose.pose.position.x, pose.pose.position.y);
            pose.header.stamp = ros::Time::now();
            pose.header.frame_id = global_frame_id_;
            pose.pose.orientation.w = 1;
            plan.push_back(pose);

            if (first){
                first = false;
                first_p = point;
            } else{
                second_p = point;
                double d = distance(first_p[0],
                                    first_p[1],
                                    second_p[0],
                                    second_p[1]);
                first_p = second_p;
                total_dist += d;
            }

            
        }
        std::cout << "-----> Path found, total_dist:" << total_dist << std::endl;

        //Print solution
        int i = 0;
        ros::Rate r(0.01);
        visualization_msgs::Marker markerD;
            markerD.action = visualization_msgs::Marker::DELETEALL;
            markerD.header.frame_id = global_frame_id_;
            vis_pub.publish( markerD );

        for(geometry_msgs::PoseStamped pos : plan) {
            visualization_msgs::Marker marker;

            marker.header.frame_id = global_frame_id_;
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

};