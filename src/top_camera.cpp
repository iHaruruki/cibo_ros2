// Copyright (c) 2025 Haruki Isono
// This software is released under the MIT License, see LICENSE.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class TopCameraNode : public rclcpp::Node {
public:
  TopCameraNode() : Node("top_camera") {
    // Initialize parameters
    declare_parameter("min_detection_confidence", 0.5);
    declare_parameter("min_tracking_confidence", 0.5);
    declare_parameter("roi_enabled", false);
    declare_parameter("roi_x", 0);
    declare_parameter("roi_y", 0);
    declare_parameter("roi_width", 400);
    declare_parameter("roi_height", 300);

    // Get parameters
    roi_enabled_ = get_parameter("roi_enabled").as_bool();
    roi_x_ = get_parameter("roi_x").as_int();
    roi_y_ = get_parameter("roi_y").as_int();
    roi_width_ = get_parameter("roi_width").as_int();
    roi_height_ = get_parameter("roi_height").as_int();

    // Setup OpenCV window
    setup_opencv_window();

    // Subscribers
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera_02/color/image_raw", 10,
        std::bind(&TopCameraNode::image_callback, this, std::placeholders::_1));

    // Publishers
    annotated_pub_ =
        create_publisher<sensor_msgs::msg::Image>("/top_camera/annotated_image", 10);
    pose_landmarks_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/top_camera/pose_landmarks", 10);
    left_hand_landmarks_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/top_camera/left_hand_landmarks", 10);
    right_hand_landmarks_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/top_camera/right_hand_landmarks", 10);

    RCLCPP_INFO(get_logger(), "Top Camera Node initialized");
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
    auto* self = static_cast<TopCameraNode*>(userdata);
    self->mouse_callback(event, x, y, flags);
  }

  void mouse_callback(int event, int x, int y, int flags) {
    if (event == cv::EVENT_LBUTTONDOWN) {
      dragging_ = true;
      start_point_ = cv::Point(x, y);
      end_point_ = cv::Point(x, y);
      RCLCPP_INFO(get_logger(), "ROI start point: (%d, %d)", x, y);
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

        RCLCPP_INFO(get_logger(), "ROI set: x=%d, y=%d, width=%d, height=%d", roi_x_, roi_y_,
                    roi_width_, roi_height_);
      }
    }
  }

  void process_image(const cv::Mat& cv_image, cv::Mat& annotated_image,
                     std::vector<float>& pose_landmarks, std::vector<float>& left_hand_landmarks,
                     std::vector<float>& right_hand_landmarks) {
    // Note: MediaPipe C++ API is not readily available
    // This is a placeholder that passes through the image
    // In production, you would integrate with MediaPipe using its Python API or
    // use alternative pose detection libraries

    annotated_image = cv_image.clone();

    // Empty landmark arrays (would be populated by MediaPipe detection)
    pose_landmarks.clear();
    left_hand_landmarks.clear();
    right_hand_landmarks.clear();
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      auto cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;

      // Process image
      cv::Mat annotated_image;
      std::vector<float> pose_landmarks;
      std::vector<float> left_hand_landmarks;
      std::vector<float> right_hand_landmarks;

      process_image(cv_image, annotated_image, pose_landmarks, left_hand_landmarks,
                    right_hand_landmarks);

      // Display image with ROI selection
      cv::Mat display_image = annotated_image.clone();

      // Draw ROI rectangle if enabled
      if (roi_enabled_ && roi_width_ > 0 && roi_height_ > 0) {
        cv::rectangle(display_image, cv::Point(roi_x_, roi_y_),
                      cv::Point(roi_x_ + roi_width_, roi_y_ + roi_height_), cv::Scalar(0, 255, 0),
                      2);
        cv::putText(display_image, "ROI", cv::Point(roi_x_, roi_y_ - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
      }

      // Draw temporary ROI during dragging
      if (dragging_ && start_point_.x >= 0 && end_point_.x >= 0) {
        cv::rectangle(display_image, start_point_, end_point_, cv::Scalar(255, 0, 0), 2);
        cv::putText(display_image, "Selecting ROI...",
                    cv::Point(start_point_.x, start_point_.y - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255, 0, 0), 2);
      }

      // Add instructions
      cv::putText(display_image, "Drag to select ROI", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  0.6, cv::Scalar(0, 255, 0), 2);

      cv::imshow("Top Camera - ROI Selection", display_image);

      // Handle key presses
      int key = cv::waitKey(1) & 0xFF;
      if (key == 'q') {
        cv::destroyAllWindows();
      } else if (key == 'r') {
        roi_enabled_ = false;
        set_parameters({rclcpp::Parameter("roi_enabled", false)});
        RCLCPP_INFO(get_logger(), "ROI reset");
      }

      // Publish results
      try {
        // Publish annotated image
        auto annotated_msg = cv_bridge::CvImage(msg->header, "bgr8", annotated_image).toImageMsg();
        annotated_pub_->publish(*annotated_msg);

        // Publish landmarks
        auto pose_msg = std_msgs::msg::Float32MultiArray();
        pose_msg.data = pose_landmarks;
        pose_landmarks_pub_->publish(pose_msg);

        auto left_hand_msg = std_msgs::msg::Float32MultiArray();
        left_hand_msg.data = left_hand_landmarks;
        left_hand_landmarks_pub_->publish(left_hand_msg);

        auto right_hand_msg = std_msgs::msg::Float32MultiArray();
        right_hand_msg.data = right_hand_landmarks;
        right_hand_landmarks_pub_->publish(right_hand_msg);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error publishing data: %s", e.what());
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pose_landmarks_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr left_hand_landmarks_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr right_hand_landmarks_pub_;

  bool dragging_ = false;
  cv::Point start_point_{-1, -1};
  cv::Point end_point_{-1, -1};

  bool roi_enabled_ = false;
  int roi_x_ = 0;
  int roi_y_ = 0;
  int roi_width_ = 400;
  int roi_height_ = 300;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TopCameraNode>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
