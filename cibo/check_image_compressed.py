#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        
        self.bridge = CvBridge()
        self.declare_parameter('show_images', True)
        
        # color/image_raw
        self.color_raw_sub = self.create_subscription(
            Image,
            '/top_camera/color/image_raw',
            self.color_raw_callback,
            10
        )
        self.get_logger().info("Subscribed to /color/image_raw")
        
        # color/image_raw/compressed
        self.color_compressed_sub = self.create_subscription(
            CompressedImage,
            '/top_camera/color/image_raw/compressed',
            self.color_compressed_callback,
            10
        )
        self.get_logger().info("Subscribed to /color/image_raw/compressed")
        
        # depth/image_raw
        self.depth_raw_sub = self.create_subscription(
            Image,
            '/top_camera/depth/image_raw',
            self.depth_raw_callback,
            10
        )
        self.get_logger().info("Subscribed to /depth/image_raw")
        
        # depth/image_raw/compressed
        self.depth_compressed_sub = self.create_subscription(
            CompressedImage,
            '/top_camera/depth/image_raw/compressedDepth',
            self.depth_compressed_callback,
            10
        )
        self.get_logger().info("Subscribed to /depth/image_raw/compressedDepth")
        
        # color/camera_info
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            '/top_camera/color/camera_info',
            self.color_info_callback,
            10
        )
        
        # depth/camera_info
        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            '/top_camera/depth/camera_info',
            self.depth_info_callback,
            10
        )
        
        self.get_logger().info("Camera Subscriber Node started")

    # color/image_raw
    def color_raw_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(
                f"🎨 Color Raw - Shape: {cv_image.shape}, Encoding: {msg.encoding}"
            )
            cv2.imshow('Color Image (RAW)', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in color_raw_callback: {e}")

    # color/image_raw/compressed
    def color_compressed_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.get_logger().info(
                f"🎨 Color Compressed - Shape: {cv_image.shape}, "
                f"Format: {msg.format}, Size: {len(msg.data)} bytes"
            )
            cv2.imshow('Color Image (COMPRESSED)', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in color_compressed_callback: {e}")

    # depth/image_raw
    def depth_raw_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            valid_depth = cv_image[cv_image > 0]
            if len(valid_depth) > 0:
                self.get_logger().info(
                    f"Depth Raw - Min: {valid_depth.min():.3f}m, "
                    f"Max: {valid_depth.max():.3f}m, Mean: {valid_depth.mean():.3f}m"
                )
            
            # デプス画像を可視化
            depth_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_color = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Depth Image (RAW)', depth_color)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in depth_raw_callback: {e}")

    # depth/image_raw/compressed
    def depth_compressed_callback(self, msg):
        try:
            # compressedDepth形式をデコード
            # ヘッダー情報を読み取る
            depth_header = msg.data[:12]
            depth_data = msg.data[12:]
            
            # PNG圧縮データをデコード
            np_arr = np.frombuffer(depth_data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
            
            if cv_image is not None:
                valid_depth = cv_image[cv_image > 0]
                if len(valid_depth) > 0:
                    self.get_logger().info(
                        f"Depth Compressed - Min: {valid_depth.min():.3f}m, "
                        f"Max: {valid_depth.max():.3f}m, Mean: {valid_depth.mean():.3f}m, "
                        f"Format: {msg.format}, Size: {len(msg.data)} bytes"
                    )
                
                # デプス画像を可視化
                depth_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_color = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
                cv2.imshow('Depth Image (COMPRESSED)', depth_color)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in depth_compressed_callback: {e}")

    # color/camera_info
    def color_info_callback(self, msg):
        self.get_logger().info(
            f"Color Camera Info - Resolution: {msg.width}x{msg.height}, "
            f"Frame ID: {msg.header.frame_id}"
        )

    # depth/camera_info
    def depth_info_callback(self, msg):
        self.get_logger().info(
            f"Depth Camera Info - Resolution: {msg.width}x{msg.height}, "
            f"Frame ID: {msg.header.frame_id}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    
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