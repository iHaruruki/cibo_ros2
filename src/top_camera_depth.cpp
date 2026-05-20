// Copyright (c) 2025 Haruki Isono
// This software is released under the MIT License, see LICENSE.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cmath>

using CompressedImageMsg = sensor_msgs::msg::CompressedImage;
using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<CompressedImageMsg, CompressedImageMsg, CameraInfoMsg>;

class TopCameraDepthNode : public rclcpp::Node {
public:
  TopCameraDepthNode() : Node("top_camera"), tf_broadcaster_(this) {
    // Declare parameters
    declare_parameter("min_detection_confidence", 0.6);
    declare_parameter("min_tracking_confidence", 0.6);
    declare_parameter("roi_enabled", false);
    declare_parameter("roi_x", 0);
    declare_parameter("roi_y", 0);
    declare_parameter("roi_width", 400);
    declare_parameter("roi_height", 300);
    declare_parameter("color_topic", "/top_camera/color/image_raw");
    declare_parameter("color_info_topic", "/top_camera/color/camera_info");
    declare_parameter("depth_topic", "/top_camera/depth/image_raw");
    declare_parameter("depth_info_topic", "/top_camera/depth/camera_info");
    declare_parameter("camera_frame", "top_camera_depth_optical_frame");
    declare_parameter("tf_rate_hz", 30.0);

    // Get parameters
    roi_enabled_ = get_parameter("roi_enabled").as_bool();
    roi_x_ = get_parameter("roi_x").as_int();
    roi_y_ = get_parameter("roi_y").as_int();
    roi_width_ = get_parameter("roi_width").as_int();
    roi_height_ = get_parameter("roi_height").as_int();
    color_topic_ = get_parameter("color_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    camera_frame_ = get_parameter("camera_frame").as_string();
    tf_rate_hz_ = get_parameter("tf_rate_hz").as_double();

    // Setup OpenCV window
    setup_opencv_window();

    // Setup synchronizer for color, depth, and camera info
    color_sub_ = std::make_unique<message_filters::Subscriber<CompressedImageMsg>>(
        this, color_topic_, rclcpp::QoS(10));
    depth_sub_ = std::make_unique<message_filters::Subscriber<CompressedImageMsg>>(
        this, depth_topic_, rclcpp::QoS(10));
    depth_info_sub_ = std::make_unique<message_filters::Subscriber<CameraInfoMsg>>(
        this, get_parameter("depth_info_topic").as_string(), rclcpp::QoS(10));

    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(20), *color_sub_, *depth_sub_, *depth_info_sub_);
    sync_->registerCallback(std::bind(&TopCameraDepthNode::synced_callback, this,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));

    // Publishers
    annotated_pub_ =
        create_publisher<sensor_msgs::msg::Image>("/top_camera/annotated_image", 10);
    left_hand_landmarks_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/top_camera/left_hand_landmarks", 10);
    right_hand_landmarks_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/top_camera/right_hand_landmarks", 10);

    last_tf_time_ = now();

    RCLCPP_INFO(get_logger(), "Top Camera Depth Node initialized (hands only, with depth→3D & tf broadcasting)");
  }

private:
  void setup_opencv_window() {
    try {
      cv::namedWindow("Top Camera - ROI Selection", cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback("Top Camera - ROI Selection", mouse_callback_static, this);
      RCLCPP_INFO(get_logger(), "OpenCV window setup for ROI selection");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to setup OpenCV window: %s", e.what());
    }
  }

  static void mouse_callback_static(int event, int x, int y, int flags, void* userdata) {
    auto* self = static_cast<TopCameraDepthNode*>(userdata);
    self->mouse_callback(event, x, y, flags);
  }

  void mouse_callback(int event, int x, int y, int flags) {
    if (event == cv::EVENT_LBUTTONDOWN) {
      dragging_ = true;
      start_point_ = cv::Point(x, y);
      end_point_ = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE && dragging_) {
      end_point_ = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP) {
      if (dragging_ && start_point_.x >= 0) {
        dragging_ = false;
        end_point_ = cv::Point(x, y);

        int x1 = std::min(start_point_.x, end_point_.x);
        int y1 = std::min(start_point_.y, end_point_.y);
        int x2 = std::max(start_point_.x, end_point_.x);
        int y2 = std::max(start_point_.y, end_point_.y);

        roi_x_ = x1;
        roi_y_ = y1;
        roi_width_ = x2 - x1;
        roi_height_ = y2 - y1;
        roi_enabled_ = true;

        set_parameters({rclcpp::Parameter("roi_enabled", true),
                        rclcpp::Parameter("roi_x", roi_x_),
                        rclcpp::Parameter("roi_y", roi_y_),
                        rclcpp::Parameter("roi_width", roi_width_),
                        rclcpp::Parameter("roi_height", roi_height_)});

        RCLCPP_INFO(get_logger(), "ROI set: x=%d, y=%d, w=%d, h=%d", roi_x_, roi_y_,
                    roi_width_, roi_height_);
      }
    }
  }

  void synced_callback(const CompressedImageMsg::SharedPtr color_msg,
                       const CompressedImageMsg::SharedPtr depth_msg,
                       const CameraInfoMsg::SharedPtr depth_info) {
    try {
      // Decode color image
      cv::Mat color;
      try {
        std::vector<uint8_t> buf(color_msg->data.begin(), color_msg->data.end());
        color = cv::imdecode(buf, cv::IMREAD_COLOR);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Color cv bridge error: %s", e.what());
        return;
      }

      // Decode depth image - skip header if compressedDepth format
      cv::Mat depth_m;
      try {
        std::vector<uint8_t> depth_data = depth_msg->data;
        if (depth_data.size() > 12) {
          // Skip the 12-byte header
          depth_data.erase(depth_data.begin(), depth_data.begin() + 12);
        }
        cv::Mat depth_encoded = cv::imdecode(depth_data, cv::IMREAD_ANYDEPTH);
        if (!depth_encoded.empty()) {
          depth_encoded.convertTo(depth_m, CV_32F);
          depth_m = depth_m / 1000.0f;  // Convert from mm to m
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth cv bridge error: %s", e.what());
        return;
      }

      // Process image (placeholder - no MediaPipe in C++)
      cv::Mat annotated_image = color.clone();
      std::vector<float> left_hand_lm;
      std::vector<float> right_hand_lm;

      // Publish annotated image
      auto ann = cv_bridge::CvImage(color_msg->header, "bgr8", annotated_image).toImageMsg();
      // Update timestamp from color message header
      ann->header.stamp = color_msg->header.stamp;
      annotated_pub_->publish(*ann);

      // Publish landmarks
      publish_array(left_hand_landmarks_pub_, left_hand_lm);
      publish_array(right_hand_landmarks_pub_, right_hand_lm);

      // Display
      cv::Mat display_image = annotated_image.clone();
      if (roi_enabled_ && roi_width_ > 0 && roi_height_ > 0) {
        cv::rectangle(display_image, cv::Point(roi_x_, roi_y_),
                      cv::Point(roi_x_ + roi_width_, roi_y_ + roi_height_), cv::Scalar(0, 255, 0),
                      2);
        cv::putText(display_image, "ROI", cv::Point(roi_x_, roi_y_ - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
      }
      if (dragging_ && start_point_.x >= 0) {
        cv::rectangle(display_image, start_point_, end_point_, cv::Scalar(255, 0, 0), 2);
        cv::putText(display_image, "Selecting ROI...",
                    cv::Point(start_point_.x, start_point_.y - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255, 0, 0), 2);
      }

      cv::putText(display_image, "Drag to select ROI", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  0.6, cv::Scalar(0, 255, 0), 2);
      cv::imshow("Top Camera - ROI Selection", display_image);

      int key = cv::waitKey(1) & 0xFF;
      if (key == 'q') {
        cv::destroyAllWindows();
      } else if (key == 'r') {
        roi_enabled_ = false;
        set_parameters({rclcpp::Parameter("roi_enabled", false)});
        RCLCPP_INFO(get_logger(), "ROI reset");
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error in callback: %s", e.what());
    }
  }

  void publish_array(rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub,
                     const std::vector<float>& data) {
    auto msg = std_msgs::msg::Float32MultiArray();
    msg.data = data;
    pub->publish(msg);
  }

  std::unique_ptr<message_filters::Subscriber<CompressedImageMsg>> color_sub_;
  std::unique_ptr<message_filters::Subscriber<CompressedImageMsg>> depth_sub_;
  std::unique_ptr<message_filters::Subscriber<CameraInfoMsg>> depth_info_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr left_hand_landmarks_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr right_hand_landmarks_pub_;

  tf2_ros::TransformBroadcaster tf_broadcaster_;

  bool dragging_ = false;
  cv::Point start_point_{-1, -1};
  cv::Point end_point_{-1, -1};

  bool roi_enabled_ = false;
  int roi_x_ = 0;
  int roi_y_ = 0;
  int roi_width_ = 400;
  int roi_height_ = 300;

  std::string color_topic_;
  std::string depth_topic_;
  std::string camera_frame_;
  double tf_rate_hz_ = 30.0;
  rclcpp::Time last_tf_time_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TopCameraDepthNode>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
