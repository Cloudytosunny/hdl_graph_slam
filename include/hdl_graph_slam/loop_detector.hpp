// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/graph_slam.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

namespace hdl_graph_slam {

struct Loop {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Loop>;

  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2, const Eigen::Matrix4f& relpose) : key1(key1), key2(key2), relative_pose(relpose) {}

public:
  KeyFrame::Ptr key1;
  KeyFrame::Ptr key2;
  Eigen::Matrix4f relative_pose;
};

/**
 * @brief this class finds loops by scam matching and adds them to the pose graph
 */
class LoopDetector {
public:
  typedef pcl::PointXYZI PointT;

  /**
   * @brief constructor  //构造器
   * @param pnh
   */
  LoopDetector(ros::NodeHandle& pnh) {
    distance_thresh = pnh.param<double>("distance_thresh", 5.0);
    accum_distance_thresh = pnh.param<double>("accum_distance_thresh", 8.0);
    distance_from_last_edge_thresh = pnh.param<double>("min_edge_interval", 5.0);

    fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
    fitness_score_thresh = pnh.param<double>("fitness_score_thresh", 0.5);

    registration = select_registration_method(pnh);
    last_edge_accum_distance = 0.0;
  }

  /**
   * @brief detect loops and add them to the pose graph
   * @param keyframes       keyframes  已有的关键帧
   * @param new_keyframes   newly registered keyframes  新记录的关键帧
   * @param graph_slam      pose graph
   */
  std::vector<Loop::Ptr> detect(const std::vector<KeyFrame::Ptr>& keyframes, const std::deque<KeyFrame::Ptr>& new_keyframes, hdl_graph_slam::GraphSLAM& graph_slam) {
    std::vector<Loop::Ptr> detected_loops;
    for(const auto& new_keyframe : new_keyframes) {
      // 在历史关键帧中寻找可能的回环(candidates)
      auto candidates = find_candidates(keyframes, new_keyframe);
      //分别将可能的回环(candidates)与new_keyframe 配准，得到检测的回环
      auto loop = matching(candidates, new_keyframe, graph_slam);
      if(loop) {
        detected_loops.push_back(loop);
      }
    }

    return detected_loops;
  }

  double get_distance_thresh() const {
    return distance_thresh;
  }

private:
  /**
   * @brief find loop candidates. A detected loop begins at one of #keyframes and ends at #new_keyframe
   * @param keyframes      candidate keyframes of loop start
   * @param new_keyframe   loop end keyframe
   * @return loop candidates
   */
  // 寻找可能的回环
  std::vector<KeyFrame::Ptr> find_candidates(const std::vector<KeyFrame::Ptr>& keyframes, const KeyFrame::Ptr& new_keyframe) const {
    // too close to the last registered loop edge  与最新的关键帧不能太近
    if(new_keyframe->accum_distance - last_edge_accum_distance < distance_from_last_edge_thresh) {
      return std::vector<KeyFrame::Ptr>();
    }

    std::vector<KeyFrame::Ptr> candidates;
    candidates.reserve(32);   //预留空间

    for(const auto& k : keyframes) {
      // traveled distance between keyframes is too small  准则1：关键帧之间运动的距离大于某一个阈值
      if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh) {
        continue;   //结束本次循环  进行下一次循环
      }

      const auto& pos1 = k->node->estimate().translation();
      const auto& pos2 = new_keyframe->node->estimate().translation();

      // estimated distance between keyframes is too small   关键帧之间位姿估计的位移小于某个阈值
      double dist = (pos1.head<2>() - pos2.head<2>()).norm();
      if(dist > distance_thresh) {
        continue;
      }

      candidates.push_back(k);
    }

    return candidates;
  }

  /**              要验证循环候选，此函数将在构成循环的关键帧之间应用扫描匹配。如果它们匹配良好，循环将添加到姿势图中
   * @brief To validate a loop candidate this function applies a scan matching between keyframes consisting the loop. If they are matched well, the loop is added to the pose graph
   * @param candidate_keyframes  candidate keyframes of loop start
   * @param new_keyframe         loop end keyframe
   * @param graph_slam           graph slam
   */
  Loop::Ptr matching(const std::vector<KeyFrame::Ptr>& candidate_keyframes, const KeyFrame::Ptr& new_keyframe, hdl_graph_slam::GraphSLAM& graph_slam) {
    if(candidate_keyframes.empty()) {
      return nullptr;
    }

    registration->setInputTarget(new_keyframe->cloud);   // pcl中点云配准算法，设置目标点云

    double best_score = std::numeric_limits<double>::max();
    KeyFrame::Ptr best_matched;
    Eigen::Matrix4f relative_pose;

    std::cout << std::endl;
    std::cout << "--- loop detection ---" << std::endl;
    std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
    std::cout << "matching" << std::flush;
    auto t1 = ros::Time::now();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());

    // 配准并得到两关键帧之间的相对位姿
    for(const auto& candidate : candidate_keyframes) {
      registration->setInputSource(candidate->cloud);
      // 利用估计的关键帧相对位姿作为配准初始值
      Eigen::Isometry3d new_keyframe_estimate = new_keyframe->node->estimate();
      new_keyframe_estimate.linear() = Eigen::Quaterniond(new_keyframe_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Isometry3d candidate_estimate = candidate->node->estimate();
      candidate_estimate.linear() = Eigen::Quaterniond(candidate_estimate.linear()).normalized().toRotationMatrix();   //变换的线性部分的可写表达式
      Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() * candidate_estimate).matrix().cast<float>();  // 变换的初始总估计
      guess(2, 3) = 0.0;
      registration->align(*aligned, guess);
      std::cout << "." << std::flush;
//  目标中一个点与其对应关系之间的最大允许距离  fitness_score_max_range
      double score = registration->getFitnessScore(fitness_score_max_range);  //获得欧几里得适应度分数（例如，从源到目标的距离平方的平均值）
      // 判断配准是否收敛且配准评分
      if(!registration->hasConverged() || score > best_score) {
        continue;
      }

      best_score = score;
      best_matched = candidate;
      relative_pose = registration->getFinalTransformation();   //得到配准方法估计的最终变换矩阵。
    }

    auto t2 = ros::Time::now();
    std::cout << " done" << std::endl;
    std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

    if(best_score > fitness_score_thresh) {
      std::cout << "loop not found..." << std::endl;
      return nullptr;
    }

    std::cout << "loop found!!" << std::endl;
    std::cout << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - " << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << std::endl;

    last_edge_accum_distance = new_keyframe->accum_distance;

    return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
  }

private:
  double distance_thresh;                 // estimated distance between keyframes consisting a loop must be less than this distance
  double accum_distance_thresh;           // traveled distance between ...
  double distance_from_last_edge_thresh;  // a new loop edge must far from the last one at least this distance

  double fitness_score_max_range;  // maximum allowable distance between corresponding points
  double fitness_score_thresh;     // threshold for scan matching

  double last_edge_accum_distance;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
