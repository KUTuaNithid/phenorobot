#ifndef OPENVSLAM_ROS_H
#define OPENVSLAM_ROS_H

#include <openvslam/system.h>
#include <openvslam/config.h>
#include <openvslam/util/stereo_rectifier.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <darknet_ros_msgs/centerBdbox.h>
#include <darknet_ros_msgs/centerBdboxes.h>
#include <darknet_ros_msgs/get_camParam.h>

namespace openvslam_ros {
class system {
public:
    system(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path);
    openvslam::system SLAM_;
    std::shared_ptr<openvslam::config> cfg_;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    std::chrono::steady_clock::time_point tp_0_;
    cv::Mat mask_;
    std::vector<double> track_times_;
};

class mono : public system {
public:
    mono(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path);
    void callback(const sensor_msgs::ImageConstPtr& msg);

    image_transport::Subscriber sub_;
};
class stereo : public system {
public:
    stereo(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path,
           const bool rectify);
    // void callback(const sensor_msgs::ImageConstPtr& left, const sensor_msgs::ImageConstPtr& right);
    void callback(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right, const darknet_ros_msgs::centerBdboxes::ConstPtr &bdbox);

    std::shared_ptr<openvslam::util::stereo_rectifier> rectifier_;
    // image_transport::SubscriberFilter left_sf_, right_sf_;
    message_filters::Subscriber<sensor_msgs::Image> left_sf_, right_sf_;
    message_filters::Subscriber<darknet_ros_msgs::centerBdboxes> bdbox_sf_;
    // using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, darknet_ros_msgs::centerBdboxes>;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, darknet_ros_msgs::centerBdboxes> SyncPolicy;
    // using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;
    // message_filters::Synchronizer<SyncPolicy> sync_;
};

class rgbd : public system {
public:
    rgbd(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path);
    void callback(const sensor_msgs::ImageConstPtr& color, const sensor_msgs::ImageConstPtr& depth);

    image_transport::SubscriberFilter color_sf_, depth_sf_;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
    message_filters::Synchronizer<SyncPolicy> sync_;
};
} // namespace openvslam_ros

#endif // OPENVSLAM_ROS_H
