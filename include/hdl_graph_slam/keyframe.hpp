// SPDX-License-Identifier: BSD-2-Clause

#ifndef KEYFRAME_HPP
#define KEYFRAME_HPP

#include <boost/optional.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

namespace g2o {
class VertexSE3;
class HyperGraph;
class SparseOptimizer;
} // namespace g2o

namespace hdl_graph_slam {

/**
 * @brief KeyFrame (pose node)
 */
struct KeyFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointT = pcl::PointXYZI;
  using Ptr = std::shared_ptr<KeyFrame>;

  KeyFrame(const ros::Time &stamp, const Eigen::Isometry3d &odom,
           double accum_distance,
           const pcl::PointCloud<PointT>::ConstPtr &cloud);
  KeyFrame(const std::string &directory, g2o::HyperGraph *graph);
  virtual ~KeyFrame();

  void save(const std::string &directory);
  bool load(const std::string &directory, g2o::HyperGraph *graph);

  long id() const;
  Eigen::Isometry3d estimate() const;

public:
  ros::Time stamp; // timestamp 时间戳
  Eigen::Isometry3d
      odom; // odometry (estimated by scan_matching_odometry) 里程计
  double accum_distance; // accumulated distance from the first node (by
                         // scan_matching_odometry)从第一个节点累计距离
  pcl::PointCloud<PointT>::ConstPtr cloud; // point cloud
  boost::optional<Eigen::Vector4d>
      floor_coeffs; // detected floor's coefficients  检测地面的系数
  boost::optional<Eigen::Vector3d>
      utm_coord; // UTM coord obtained by GPS 通过GPS 获取UTM坐标

  boost::optional<Eigen::Vector3d> acceleration;   //
  boost::optional<Eigen::Quaterniond> orientation; //

  g2o::VertexSE3 *node; // node instance
};

/**
 * @brief KeyFramesnapshot for map cloud generation
 */
struct KeyFrameSnapshot {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointT = KeyFrame::PointT;
  using Ptr = std::shared_ptr<KeyFrameSnapshot>;

  KeyFrameSnapshot(const KeyFrame::Ptr &key);
  KeyFrameSnapshot(const Eigen::Isometry3d &pose,
                   const pcl::PointCloud<PointT>::ConstPtr &cloud);

  ~KeyFrameSnapshot();

public:
  Eigen::Isometry3d pose; // pose estimated by graph optimization
  pcl::PointCloud<PointT>::ConstPtr cloud; // point cloud
};

} // namespace hdl_graph_slam

#endif // KEYFRAME_HPP
