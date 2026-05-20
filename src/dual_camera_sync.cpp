// Copyright (c) 2025 Haruki Isono
// This software is released under the MIT License, see LICENSE.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using ImageMsg = sensor_msgs::msg::Image;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg>;

class DualCameraSyncNode : public rclcpp::Node {
public:
  DualCameraSyncNode() : Node("dual_camera_sync") {
    // Create QoS profile for best effort
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

    // Subscribe to camera topics
    front_color_sub_ = std::make_unique<message_filters::Subscriber<ImageMsg>>(
        this, "/front_camera/color/image_raw", qos_profile);
    front_depth_sub_ = std::make_unique<message_filters::Subscriber<ImageMsg>>(
        this, "/front_camera/depth/image_raw", qos_profile);
    top_color_sub_ = std::make_unique<message_filters::Subscriber<ImageMsg>>(
        this, "/top_camera/color/image_raw", qos_profile);
    top_depth_sub_ = std::make_unique<message_filters::Subscriber<ImageMsg>>(
        this, "/top_camera/depth/image_raw", qos_profile);

    // Setup synchronizer
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(50), *front_color_sub_, *front_depth_sub_, *top_color_sub_, *top_depth_sub_);
    sync_->registerCallback(
        std::bind(&DualCameraSyncNode::camera_callback, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    // Create OpenCV windows
    cv::namedWindow("FRONT Camera Color", cv::WINDOW_NORMAL);
    cv::namedWindow("FRONT Camera Depth", cv::WINDOW_NORMAL);
    cv::namedWindow("TOP Camera Color", cv::WINDOW_NORMAL);
    cv::namedWindow("TOP Camera Depth", cv::WINDOW_NORMAL);

    RCLCPP_INFO(get_logger(), "Dual Camera Sync Node initialized");
  }

private:
  void camera_callback(const ImageMsg::SharedPtr front_color_msg,
                       const ImageMsg::SharedPtr front_depth_msg,
                       const ImageMsg::SharedPtr top_color_msg,
                       const ImageMsg::SharedPtr top_depth_msg) {
    try {
      // Convert front color image
      auto front_color_cv = cv_bridge::toCvShare(front_color_msg, "bgr8")->image;
      cv::imshow("FRONT Camera Color", front_color_cv);

      // Convert front depth image
      auto front_depth_cv = cv_bridge::toCvShare(front_depth_msg, "passthrough")->image;
      cv::Mat front_depth_normalized;
      cv::normalize(front_depth_cv, front_depth_normalized, 0, 255, cv::NORM_MINMAX);
      cv::Mat front_depth_8bit;
      front_depth_normalized.convertTo(front_depth_8bit, CV_8U);
      cv::Mat front_depth_colored;
      cv::applyColorMap(front_depth_8bit, front_depth_colored, cv::COLORMAP_JET);
      cv::imshow("FRONT Camera Depth", front_depth_colored);

      // Convert top color image
      auto top_color_cv = cv_bridge::toCvShare(top_color_msg, "bgr8")->image;
      cv::imshow("TOP Camera Color", top_color_cv);

      // Convert top depth image
      auto top_depth_cv = cv_bridge::toCvShare(top_depth_msg, "passthrough")->image;
      cv::Mat top_depth_normalized;
      cv::normalize(top_depth_cv, top_depth_normalized, 0, 255, cv::NORM_MINMAX);
      cv::Mat top_depth_8bit;
      top_depth_normalized.convertTo(top_depth_8bit, CV_8U);
      cv::Mat top_depth_colored;
      cv::applyColorMap(top_depth_8bit, top_depth_colored, cv::COLORMAP_JET);
      cv::imshow("TOP Camera Depth", top_depth_colored);

      cv::waitKey(1);
    } catch (std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error processing camera images: %s", e.what());
    }
  }

  std::unique_ptr<message_filters::Subscriber<ImageMsg>> front_color_sub_;
  std::unique_ptr<message_filters::Subscriber<ImageMsg>> front_depth_sub_;
  std::unique_ptr<message_filters::Subscriber<ImageMsg>> top_color_sub_;
  std::unique_ptr<message_filters::Subscriber<ImageMsg>> top_depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DualCameraSyncNode>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
