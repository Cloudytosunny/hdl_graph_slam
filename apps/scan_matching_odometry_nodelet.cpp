// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/ScanMatchingStatus.h>

namespace hdl_graph_slam {

class ScanMatchingOdometryNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  //为了解决使用Eigen的对象，可能出现堆栈对齐的问题

  ScanMatchingOdometryNodelet() {}
  virtual ~ScanMatchingOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    if(private_nh.param<bool>("enable_imu_frontend", false)) {
      msf_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, false));
      msf_pose_after_update_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose_after_update", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, true));
    }

    points_sub = nh.subscribe("/filtered_points", 256, &ScanMatchingOdometryNodelet::cloud_callback, this);
    read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 32);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 32);
    trans_pub = nh.advertise<geometry_msgs::TransformStamped>("/scan_matching_odometry/transform", 32);
    status_pub = private_nh.advertise<ScanMatchingStatus>("/scan_matching_odometry/status", 8);
    aligned_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    points_topic = pnh.param<std::string>("points_topic", "/velodyne_points");
    odom_frame_id = pnh.param<std::string>("odom_frame_id", "odom");
    robot_odom_frame_id = pnh.param<std::string>("robot_odom_frame_id", "robot_odom");

    // The minimum tranlational distance and rotation angle between keyframes.关键帧之间的最小平移距离和旋转角度。
    // If this value is zero, frames are always compared with the previous frame 如果该值为零，则帧始终与前一帧进行比较
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding  基于阈值的注册验证
    transform_thresholding = pnh.param<bool>("transform_thresholding", false);
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }

    registration = select_registration_method(pnh);
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);
    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }

  void msf_pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg, bool after_update) {
    if(after_update) {
      msf_pose_after_update = pose_msg;
    } else {
      msf_pose = pose_msg;
    }
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud  估计输入云和关键帧云之间的相对姿势
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud  输入云和关键帧云之间的相对姿势
   */
  Eigen::Matrix4f matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    //还没有关键帧，说明它是第一帧，则直接把它设置成关键帧，并初始化位姿为单位阵，直接返回
    if(!keyframe) {
      prev_time = ros::Time();
      prev_trans.setIdentity();   //单位矩阵
      keyframe_pose.setIdentity();
      keyframe_stamp = stamp;
      keyframe = downsample(cloud);
      registration->setInputTarget(keyframe);  //提供目标点云数据集（必须包含XYZ数据！），用于计算对应距离。
      return Eigen::Matrix4f::Identity();
    }
  //当前点云给registration
    auto filtered = downsample(cloud);
    registration->setInputSource(filtered);  //pcl库里面的函数   计算对应距离
//执行匹配
    std::string msf_source;
    Eigen::Isometry3f msf_delta = Eigen::Isometry3f::Identity();  //得到一个4x4的单位矩阵

    if(private_nh.param<bool>("enable_imu_frontend", false)) {
      if(msf_pose && msf_pose->header.stamp > keyframe_stamp && msf_pose_after_update && msf_pose_after_update->header.stamp > keyframe_stamp) {
        Eigen::Isometry3d pose0 = pose2isometry(msf_pose_after_update->pose.pose);
        Eigen::Isometry3d pose1 = pose2isometry(msf_pose->pose.pose);
        Eigen::Isometry3d delta = pose0.inverse() * pose1;

        msf_source = "imu";
        msf_delta = delta.cast<float>();
      } else {
        std::cerr << "msf data is too old" << std::endl;
      }
    } else if(private_nh.param<bool>("enable_robot_odometry_init_guess", false) && !prev_time.isZero()) {
      tf::StampedTransform transform;
      if(tf_listener.waitForTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
        tf_listener.lookupTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
      } else if(tf_listener.waitForTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
        tf_listener.lookupTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
      }

      if(transform.stamp_.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        msf_source = "odometry";
        msf_delta = tf2isometry(transform).cast<float>();
      }
    }

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->align(*aligned, prev_trans * msf_delta.matrix());  //估计转换并将转换后的源（输入）作为输出返回。  输出：得到的输入转换点云数据集（精准估计）
// 匹配结束、则返回它相对于关键帧位姿？    prev_trans : 基于关键帧的上一次估计变换                                                                输入 ：转换的初始粗略估计
    publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

    if(!registration->hasConverged()) {    //是否收敛
      NODELET_INFO_STREAM("scan matching has not converged!!");
      NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
      return keyframe_pose * prev_trans;  //为什么返回上一帧
    }

    Eigen::Matrix4f trans = registration->getFinalTransformation();  //通过配准方法得到最终的变换矩阵。
    Eigen::Matrix4f odom = keyframe_pose * trans;  //应该是  世界坐标


    //这段代码的主要功能是判断当前匹配的相对位姿是否过大，载体不可能短时间内有这么大的运动
    //所以如果出现了，就说明这一帧匹配可能不正常,则直接舍弃
    if(transform_thresholding) {                               //转换阈值
      Eigen::Matrix4f delta = prev_trans.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

      if(dx > max_acceptable_trans || da > max_acceptable_angle) {
        NODELET_INFO_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
        NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
        return keyframe_pose * prev_trans;
      }
    }

    prev_time = stamp;
    prev_trans = trans;

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    keyframe_broadcaster.sendTransform(keyframe_trans);


    //判断这一帧相对于关键帧的位置或角度是否比较大了
    //如果比较大，则需要更新关键帧，并把新的关键帧的点云设置成Target
    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp).toSec();
    if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
      keyframe = filtered;
      registration->setInputTarget(keyframe);

      keyframe_pose = odom;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans.setIdentity();
    }

    if (aligned_points_pub.getNumSubscribers() > 0)
    {
      pcl::transformPointCloud (*cloud, *aligned, odom);
      aligned->header.frame_id=odom_frame_id;
      aligned_points_pub.publish(*aligned);
    }

    return odom;
  }

  /**
   * @brief publish odometry   包含位置及速度
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& base_frame_id, const Eigen::Matrix4f& pose) {
    // publish transform stamped for IMU integration
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, base_frame_id);
    trans_pub.publish(odom_trans);

    // broadcast the transform over tf
    odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = base_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    odom_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status
   */
  void publish_scan_matching_status(const ros::Time& stamp, const std::string& frame_id, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned, const std::string& msf_source, const Eigen::Isometry3f& msf_delta) {
    if(!status_pub.getNumSubscribers()) {
      return;
    }

    ScanMatchingStatus status;
    status.header.frame_id = frame_id;
    status.header.stamp = stamp;
    status.has_converged = registration->hasConverged();  //在最后一次对齐运行后返回收敛状态。
    status.matching_error = registration->getFitnessScore();    //获得欧几里得适应度分数

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    status.relative_pose = isometry2pose(Eigen::Isometry3f(registration->getFinalTransformation()).cast<double>());

    if(!msf_source.empty()) {
      status.prediction_labels.resize(1);
      status.prediction_labels[0].data = msf_source;

      status.prediction_errors.resize(1);
      Eigen::Isometry3f error = Eigen::Isometry3f(registration->getFinalTransformation()).inverse() * msf_delta;
      status.prediction_errors[0] = isometry2pose(error.cast<double>());
    }

    status_pub.publish(status);
  }

private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  ros::Subscriber points_sub;
  ros::Subscriber msf_pose_sub;
  ros::Subscriber msf_pose_after_update_sub;

  ros::Publisher odom_pub;
  ros::Publisher trans_pub;
  ros::Publisher aligned_points_pub;
  ros::Publisher status_pub;
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster odom_broadcaster;
  tf::TransformBroadcaster keyframe_broadcaster;

  std::string points_topic;
  std::string odom_frame_id;
  std::string robot_odom_frame_id;
  ros::Publisher read_until_pub;

  // keyframe parameters
  double keyframe_delta_trans;  // minimum distance between keyframes
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;

  // odometry calculation  里程计计算   带有时间标签和参考坐标的估计位姿
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose;
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose_after_update;

  ros::Time prev_time;
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe 基于关键帧的上一次估计变换
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  ros::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
