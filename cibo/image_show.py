# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

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

        # FRONTカメラのメッセージ同期
        self.front_sync = ApproximateTimeSynchronizer(
            [front_color_sub, front_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.front_sync.registerCallback(self.front_camera_callback)

        # TOPカメラのメッセージ同期
        self.top_sync = ApproximateTimeSynchronizer(
            [top_color_sub, top_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.top_sync.registerCallback(self.top_camera_callback)

        # ウィンドウ名を設定
        cv2.namedWindow("FRONT Camera Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("FRONT Camera Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera Depth", cv2.WINDOW_NORMAL)

    def front_camera_callback(self, color_msg, depth_msg):
        try:
            # FRONTカメラのカラー画像処理
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv2.imshow("FRONT Camera Color", color_image)
            
            # FRONTカメラの深度画像処理
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)
            depth_image_colored = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
            cv2.imshow("FRONT Camera Depth", depth_image_colored)
            
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing FRONT camera images: {e}")

    def top_camera_callback(self, color_msg, depth_msg):
        try:
            # TOPカメラのカラー画像処理
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv2.imshow("TOP Camera Color", color_image)
            
            # TOPカメラの深度画像処理
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)
            depth_image_colored = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
            cv2.imshow("TOP Camera Depth", depth_image_colored)
            
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing TOP camera images: {e}")

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