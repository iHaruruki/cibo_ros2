#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class CameraImageDisplay(Node):
    def __init__(self):
        super().__init__('camera_image_display')

        # CvBridgeのインスタンスを作成
        self.bridge = CvBridge()

        # QoS設定をbest_effortに変更
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # FRONTカメラの画像トピックをサブスクライブ
        front_color_sub = Subscriber(self, Image, '/front_camera/color/image_raw', qos_profile=qos_profile)
        front_depth_sub = Subscriber(self, Image, '/front_camera/depth/image_raw', qos_profile=qos_profile)
        
        # TOPカメラの画像トピックをサブスクライブ
        top_color_sub = Subscriber(self, Image, '/top_camera/color/image_raw', qos_profile=qos_profile)
        top_depth_sub = Subscriber(self, Image, '/top_camera/depth/image_raw', qos_profile=qos_profile)

        # 全カメラのメッセージ同期（FRONT color, FRONT depth, TOP color, TOP depth）
        self.camera_sync = ApproximateTimeSynchronizer(
            [front_color_sub, front_depth_sub, top_color_sub, top_depth_sub],
            queue_size=50,
            slop=0.5
        )
        self.camera_sync.registerCallback(self.camera_callback)

        # ウィンドウ名を設定
        cv2.namedWindow("FRONT Camera Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("FRONT Camera Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera Depth", cv2.WINDOW_NORMAL)

    def camera_callback(self, front_color_msg, front_depth_msg, top_color_msg, top_depth_msg):
        try:
            # FRONTカメラのカラー画像処理
            front_color_image = self.bridge.imgmsg_to_cv2(front_color_msg, desired_encoding='bgr8')
            cv2.imshow("FRONT Camera Color", front_color_image)
            
            # FRONTカメラの深度画像処理
            front_depth_image = self.bridge.imgmsg_to_cv2(front_depth_msg, desired_encoding='passthrough')
            front_depth_normalized = cv2.normalize(front_depth_image, None, 0, 255, cv2.NORM_MINMAX)
            front_depth_8bit = cv2.convertScaleAbs(front_depth_normalized)
            front_depth_colored = cv2.applyColorMap(front_depth_8bit, cv2.COLORMAP_JET)
            cv2.imshow("FRONT Camera Depth", front_depth_colored)

            # TOPカメラのカラー画像処理
            top_color_image = self.bridge.imgmsg_to_cv2(top_color_msg, desired_encoding='bgr8')
            cv2.imshow("TOP Camera Color", top_color_image)
            
            # TOPカメラの深度画像処理
            top_depth_image = self.bridge.imgmsg_to_cv2(top_depth_msg, desired_encoding='passthrough')
            top_depth_normalized = cv2.normalize(top_depth_image, None, 0, 255, cv2.NORM_MINMAX)
            top_depth_8bit = cv2.convertScaleAbs(top_depth_normalized)
            top_depth_colored = cv2.applyColorMap(top_depth_8bit, cv2.COLORMAP_JET)
            cv2.imshow("TOP Camera Depth", top_depth_colored)
            
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing camera images: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraImageDisplay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()