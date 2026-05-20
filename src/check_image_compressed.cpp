// Copyright (c) 2025 Haruki Isono
// This software is released under the MIT License, see LICENSE.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cmath>

class CameraSubscriberNode : public rclcpp::Node {
public:
  CameraSubscriberNode() : Node("camera_subscriber_node") {
    declare_parameter("show_images", true);

    // Subscribe to /color/image_raw
    color_raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/top_camera/color/image_raw", 10,
        std::bind(&CameraSubscriberNode::color_raw_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /color/image_raw");

    // Subscribe to /color/image_raw/compressed
    color_compressed_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/top_camera/color/image_raw/compressed", 10,
        std::bind(&CameraSubscriberNode::color_compressed_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /color/image_raw/compressed");

    // Subscribe to /depth/image_raw
    depth_raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/top_camera/depth/image_raw", 10,
        std::bind(&CameraSubscriberNode::depth_raw_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /depth/image_raw");

    // Subscribe to /depth/image_raw/compressedDepth
    depth_compressed_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/top_camera/depth/image_raw/compressedDepth", 10,
        std::bind(&CameraSubscriberNode::depth_compressed_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /depth/image_raw/compressedDepth");

    // Subscribe to /color/camera_info
    color_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/top_camera/color/camera_info", 10,
        std::bind(&CameraSubscriberNode::color_info_callback, this, std::placeholders::_1));

    // Subscribe to /depth/camera_info
    depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/top_camera/depth/camera_info", 10,
        std::bind(&CameraSubscriberNode::depth_info_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Camera Subscriber Node started");
  }

private:
  void color_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      auto cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
      RCLCPP_INFO(get_logger(), "Color Raw - Shape: [%d x %d], Encoding: %s", cv_image.rows,
                  cv_image.cols, msg->encoding.c_str());
      cv::imshow("/color/image_raw", cv_image);
      cv::waitKey(1);
    } catch (std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error in color_raw_callback: %s", e.what());
    }
  }

  void color_compressed_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
      cv::Mat cv_image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
      if (!cv_image.empty()) {
        RCLCPP_INFO(get_logger(), "Color Compressed - Shape: [%d x %d], Format: %s, Size: %zu bytes",
                    cv_image.rows, cv_image.cols, msg->format.c_str(), msg->data.size());
        cv::imshow("/color/image_raw/compressed", cv_image);
        cv::waitKey(1);
      }
    } catch (std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error in color_compressed_callback: %s", e.what());
    }
  }

  void depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      auto cv_image = cv_bridge::toCvShare(msg, "32FC1")->image;

      // Calculate depth statistics
      std::vector<float> valid_depths;
      for (int i = 0; i < cv_image.rows; i++) {
        for (int j = 0; j < cv_image.cols; j++) {
          float val = cv_image.at<float>(i, j);
          if (val > 0 && std::isfinite(val)) {
            valid_depths.push_back(val);
          }
        }
      }

      if (!valid_depths.empty()) {
        float min_depth = *std::min_element(valid_depths.begin(), valid_depths.end());
        float max_depth = *std::max_element(valid_depths.begin(), valid_depths.end());
        float mean_depth = 0;
        for (float v : valid_depths) {
          mean_depth += v;
        }
        mean_depth /= valid_depths.size();

        RCLCPP_INFO(get_logger(), "Depth Raw - Min: %.3f m, Max: %.3f m, Mean: %.3f m", min_depth,
                    max_depth, mean_depth);
      }

      // Visualize depth image
      cv::Mat depth_normalized;
      cv::normalize(cv_image, depth_normalized, 0, 255, cv::NORM_MINMAX);
      cv::Mat depth_8bit;
      depth_normalized.convertTo(depth_8bit, CV_8U);
      cv::Mat depth_colored;
      cv::applyColorMap(depth_8bit, depth_colored, cv::COLORMAP_JET);
      cv::imshow("/depth/image_raw", depth_colored);
      cv::waitKey(1);
    } catch (std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error in depth_raw_callback: %s", e.what());
    }
  }

  void depth_compressed_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
      // Skip the 12-byte header for compressedDepth format
      if (msg->data.size() > 12) {
        std::vector<uint8_t> depth_data(msg->data.begin() + 12, msg->data.end());
        cv::Mat cv_image = cv::imdecode(cv::Mat(depth_data), cv::IMREAD_ANYDEPTH);

        if (!cv_image.empty()) {
          // Calculate depth statistics
          std::vector<uint16_t> valid_depths;
          for (int i = 0; i < cv_image.rows; i++) {
            for (int j = 0; j < cv_image.cols; j++) {
              uint16_t val = cv_image.at<uint16_t>(i, j);
              if (val > 0) {
                valid_depths.push_back(val);
              }
            }
          }

          if (!valid_depths.empty()) {
            float min_depth = (*std::min_element(valid_depths.begin(), valid_depths.end())) / 1000.0f;
            float max_depth = (*std::max_element(valid_depths.begin(), valid_depths.end())) / 1000.0f;
            float sum_depth = 0;
            for (uint16_t v : valid_depths) {
              sum_depth += v;
            }
            float mean_depth = (sum_depth / valid_depths.size()) / 1000.0f;

            RCLCPP_INFO(get_logger(),
                        "Depth Compressed - Min: %.3f m, Max: %.3f m, Mean: %.3f m, Format: %s, "
                        "Size: %zu bytes",
                        min_depth, max_depth, mean_depth, msg->format.c_str(), msg->data.size());
          }

          // Visualize depth image
          cv::Mat depth_normalized;
          cv::normalize(cv_image, depth_normalized, 0, 255, cv::NORM_MINMAX);
          cv::Mat depth_8bit;
          depth_normalized.convertTo(depth_8bit, CV_8U);
          cv::Mat depth_colored;
          cv::applyColorMap(depth_8bit, depth_colored, cv::COLORMAP_TURBO);
          cv::imshow("/depth/image_raw/compressed", depth_colored);
          cv::waitKey(1);
        }
      }
    } catch (std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error in depth_compressed_callback: %s", e.what());
    }
  }

  void color_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Color Camera Info - Resolution: %d x %d, Frame ID: %s", msg->width,
                msg->height, msg->header.frame_id.c_str());
  }

  void depth_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Depth Camera Info - Resolution: %d x %d, Frame ID: %s", msg->width,
                msg->height, msg->header.frame_id.c_str());
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_raw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_compressed_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_raw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr depth_compressed_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraSubscriberNode>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
