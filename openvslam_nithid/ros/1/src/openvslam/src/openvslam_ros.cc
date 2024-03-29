#include <openvslam_ros.h>

#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace openvslam_ros {
system::system(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : SLAM_(cfg, vocab_file_path), cfg_(cfg), it_(nh_), tp_0_(std::chrono::steady_clock::now()),
      mask_(mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE)) {}

mono::mono(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : system(cfg, vocab_file_path, mask_img_path) {
    sub_ = it_.subscribe("camera/image_raw", 1, &mono::callback, this);
}
void mono::callback(const sensor_msgs::ImageConstPtr& msg) {
    const auto tp_1 = std::chrono::steady_clock::now();
    const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0_).count();

    // input the current frame and estimate the camera pose
    SLAM_.feed_monocular_frame(cv_bridge::toCvShare(msg)->image, timestamp, mask_);

    const auto tp_2 = std::chrono::steady_clock::now();

    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
    track_times_.push_back(track_time);
}

stereo::stereo(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path,
               const bool rectify)
    : system(cfg, vocab_file_path, mask_img_path),
      rectifier_(rectify ? std::make_shared<openvslam::util::stereo_rectifier>(cfg) : nullptr),
      left_sf_(nh_, "camera/left/image_raw", 1),
      right_sf_(nh_, "camera/right/image_raw", 1),
      bdbox_sf_(nh_, "camera/boundingbox", 1) {
    sync_.reset(new Sync(SyncPolicy(10), left_sf_, right_sf_, bdbox_sf_));
    //   sync_(SyncPolicy(10), left_sf_, right_sf_, bdbox_sf_) {
    //   left_sf_(nh_, "camera/left/image_raw", 1),
    //   right_sf_(nh_, "camera/right/image_raw", 1),
    //   sync_(SyncPolicy(10), left_sf_, right_sf_) {
    // sync_.registerCallback(&stereo::callback, this);
    sync_->registerCallback(boost::bind(&stereo::callback, this, _1, _2, _3));
}

void stereo::callback(const sensor_msgs::ImageConstPtr& left, const sensor_msgs::ImageConstPtr& right, const darknet_ros_msgs::centerBdboxes::ConstPtr &bdbox) {
// void stereo::callback(const sensor_msgs::ImageConstPtr& left, const sensor_msgs::ImageConstPtr& right) {
    auto leftcv = cv_bridge::toCvShare(left)->image;
    auto rightcv = cv_bridge::toCvShare(right)->image;
    if (leftcv.empty() || rightcv.empty()) {
        return;
    }

    if (rectifier_) {
        rectifier_->rectify(leftcv, rightcv, leftcv, rightcv);
    }

    const auto tp_1 = std::chrono::steady_clock::now();
    const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0_).count();

    // Transform bdbox to openvslam bdbox
    openvslam::data::objectdetection objects;
    
    for(unsigned int i = 0; i < bdbox->centerBdboxes.size(); i++){
        objects.add_object(bdbox->centerBdboxes[i].probability, bdbox->centerBdboxes[i].x_cen, bdbox->centerBdboxes[i].y_cen, bdbox->centerBdboxes[i].width, bdbox->centerBdboxes[i].height, bdbox->centerBdboxes[i].id, bdbox->centerBdboxes[i].Class, bdbox->centerBdboxes[i].depth);
        // objects.add_object(bdbox->centerBdboxes[i].probability, bdbox->centerBdboxes[i].x_cen, bdbox->centerBdboxes[i].y_cen, bdbox->centerBdboxes[i].width, bdbox->centerBdboxes[i].height, bdbox->centerBdboxes[i].id, bdbox->centerBdboxes[i].Class);
        // if (bdbox->centerBdboxes[i].depth != -1)
        //     ROS_INFO("bdbox->centerBdboxes[i].Class %s bdbox->centerBdboxes[i].depth %f", bdbox->centerBdboxes[i].Class.c_str(), bdbox->centerBdboxes[i].depth);
    }
    
    // input the current frame and estimate the camera pose
    // SLAM_.feed_stereo_frame(leftcv, rightcv, timestamp, mask_);
    SLAM_.feed_stereo_frame(leftcv, rightcv, timestamp, mask_, objects);

    const auto tp_2 = std::chrono::steady_clock::now();

    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
    track_times_.push_back(track_time);
}

rgbd::rgbd(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : system(cfg, vocab_file_path, mask_img_path),
      color_sf_(it_, "camera/color/image_raw", 1),
      depth_sf_(it_, "camera/depth/image_raw", 1),
      sync_(SyncPolicy(10), color_sf_, depth_sf_) {
    sync_.registerCallback(&rgbd::callback, this);
}

void rgbd::callback(const sensor_msgs::ImageConstPtr& color, const sensor_msgs::ImageConstPtr& depth) {
    auto colorcv = cv_bridge::toCvShare(color)->image;
    auto depthcv = cv_bridge::toCvShare(depth)->image;
    if (colorcv.empty() || depthcv.empty()) {
        return;
    }

    const auto tp_1 = std::chrono::steady_clock::now();
    const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0_).count();

    // input the current frame and estimate the camera pose
    SLAM_.feed_RGBD_frame(colorcv, depthcv, timestamp, mask_);

    const auto tp_2 = std::chrono::steady_clock::now();

    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
    track_times_.push_back(track_time);
}
} // namespace openvslam_ros
