// Copyright (c) 2025 Haruki Isono
// This software is released under the MIT License, see LICENSE.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ImageShowNode : public rclcpp::Node {
public:
  ImageShowNode() : Node("image_show") {
    // Subscribe to annotated images
    top_annotated_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/top_camera/annotated_image", 10,
        std::bind(&ImageShowNode::top_image_callback, this, std::placeholders::_1));

    front_annotated_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/front_camera/annotated_image", 10,
        std::bind(&ImageShowNode::front_image_callback, this, std::placeholders::_1));

    // Create OpenCV windows
    cv::namedWindow("Top Camera Annotated", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Front Camera Annotated", cv::WINDOW_AUTOSIZE);

    RCLCPP_INFO(get_logger(), "Image Show Node initialized");
  }

private:
  void top_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      auto cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
      cv::imshow("Top Camera Annotated", cv_image);
      cv::waitKey(1);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  void front_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      auto cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
      cv::imshow("Front Camera Annotated", cv_image);
      cv::waitKey(1);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr top_annotated_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr front_annotated_sub_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageShowNode>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
