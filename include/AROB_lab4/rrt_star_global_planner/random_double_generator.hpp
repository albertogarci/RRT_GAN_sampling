#ifndef RRT_STAR_GLOBAL_PLANNER_RANDOM_DOUBLE_GENERATOR_HPP_  
#define RRT_STAR_GLOBAL_PLANNER_RANDOM_DOUBLE_GENERATOR_HPP_

#include <random>
#include <cfloat>  

namespace rrt_star_global_planner {


class RandomDoubleGenerator {
 private:
  std::random_device rd_;
  double min_value_{-1.0};
  double max_value_{1.0};

 public:
  RandomDoubleGenerator() = default;

  void setRange(double min, double max);

  double generate();
};
} 

#endif 
