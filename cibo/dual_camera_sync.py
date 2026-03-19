# unified_camera_node.py
# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import message_filters
from rclpy.parameter import Parameter

class UnifiedCameraNode(Node):
    def __init__(self):
        super().__init__('unified_camera')

        # ==== CV Bridge ====
        self.bridge = CvBridge()

        # ==== QoS Profile (best_effort) ====
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==== MediaPipe ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        # ==== Parameters for Camera 01 (Front) ====
        self.declare_parameter('camera_01_min_detection_confidence', 0.6)
        self.declare_parameter('camera_01_min_tracking_confidence', 0.6)
        self.declare_parameter('camera_01_roi_enabled', False)
        self.declare_parameter('camera_01_roi_x', 0)
        self.declare_parameter('camera_01_roi_y', 0)
        self.declare_parameter('camera_01_roi_width', 400)
        self.declare_parameter('camera_01_roi_height', 300)
        self.declare_parameter('camera_01_camera_frame', 'camera_01_depth_optical_frame')
        self.declare_parameter('camera_01_publish_face_tf', False)
        self.declare_parameter('camera_01_tf_rate_hz', 30.0)
        self.declare_parameter('camera_01_overlay_alpha', 0.3)

        # ==== Parameters for Camera 02 (Top) ====
        self.declare_parameter('camera_02_min_detection_confidence', 0.6)
        self.declare_parameter('camera_02_min_tracking_confidence', 0.6)
        self.declare_parameter('camera_02_roi_enabled', False)
        self.declare_parameter('camera_02_roi_x', 0)
        self.declare_parameter('camera_02_roi_y', 0)
        self.declare_parameter('camera_02_roi_width', 400)
        self.declare_parameter('camera_02_roi_height', 300)
        self.declare_parameter('camera_02_camera_frame', 'camera_02_depth_optical_frame')
        self.declare_parameter('camera_02_tf_rate_hz', 30.0)
        self.declare_parameter('camera_02_overlay_alpha', 0.3)

        # Read Camera 01 Parameters
        self.cam01_config = {
            'min_detection_confidence': float(self.get_parameter('camera_01_min_detection_confidence').value),
            'min_tracking_confidence': float(self.get_parameter('camera_01_min_tracking_confidence').value),
            'roi_enabled': bool(self.get_parameter('camera_01_roi_enabled').value),
            'roi_x': int(self.get_parameter('camera_01_roi_x').value),
            'roi_y': int(self.get_parameter('camera_01_roi_y').value),
            'roi_width': int(self.get_parameter('camera_01_roi_width').value),
            'roi_height': int(self.get_parameter('camera_01_roi_height').value),
            'camera_frame': self.get_parameter('camera_01_camera_frame').value,
            'publish_face_tf': bool(self.get_parameter('camera_01_publish_face_tf').value),
            'tf_rate_hz': float(self.get_parameter('camera_01_tf_rate_hz').value),
            'overlay_alpha': float(self.get_parameter('camera_01_overlay_alpha').value),
        }

        # Read Camera 02 Parameters
        self.cam02_config = {
            'min_detection_confidence': float(self.get_parameter('camera_02_min_detection_confidence').value),
            'min_tracking_confidence': float(self.get_parameter('camera_02_min_tracking_confidence').value),
            'roi_enabled': bool(self.get_parameter('camera_02_roi_enabled').value),
            'roi_x': int(self.get_parameter('camera_02_roi_x').value),
            'roi_y': int(self.get_parameter('camera_02_roi_y').value),
            'roi_width': int(self.get_parameter('camera_02_roi_width').value),
            'roi_height': int(self.get_parameter('camera_02_roi_height').value),
            'camera_frame': self.get_parameter('camera_02_camera_frame').value,
            'tf_rate_hz': float(self.get_parameter('camera_02_tf_rate_hz').value),
            'overlay_alpha': float(self.get_parameter('camera_02_overlay_alpha').value),
        }

        # ==== MediaPipe Initializations ====
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.cam01_config['min_detection_confidence'],
            min_tracking_confidence=self.cam01_config['min_tracking_confidence']
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.cam01_config['min_detection_confidence'],
            min_tracking_confidence=self.cam01_config['min_tracking_confidence']
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.cam02_config['min_detection_confidence'],
            min_tracking_confidence=self.cam02_config['min_tracking_confidence']
        )

        # ==== ROI GUI State ====
        self.cam01_dragging = False
        self.cam01_start_point = None
        self.cam01_end_point = None
        self.cam02_dragging = False
        self.cam02_start_point = None
        self.cam02_end_point = None

        self.setup_opencv_windows()

        # ==== Publishers ====
        # Camera 01 (Front)
        self.cam01_annotated_pub = self.create_publisher(Image, '/front_camera/annotated_image', self.qos_profile)
        self.cam01_overlay_pub = self.create_publisher(Image, '/front_camera/overlay_image', self.qos_profile)
        self.cam01_pose_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/pose_landmarks', self.qos_profile)
        self.cam01_face_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/face_landmarks', self.qos_profile)
        self.cam01_left_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/left_hand_landmarks', self.qos_profile)
        self.cam01_right_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/right_hand_landmarks', self.qos_profile)

        # Camera 02 (Top)
        self.cam02_annotated_pub = self.create_publisher(Image, '/top_camera/annotated_image', self.qos_profile)
        self.cam02_overlay_pub = self.create_publisher(Image, '/top_camera/overlay_image', self.qos_profile)
        self.cam02_left_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/top_camera/left_hand_landmarks', self.qos_profile)
        self.cam02_right_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/top_camera/right_hand_landmarks', self.qos_profile)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)
        self.cam01_last_tf_time = self.get_clock().now()
        self.cam02_last_tf_time = self.get_clock().now()

        # ==== Subscribers with message synchronization ====
        # Camera 01 subscribers (compressed topics, best_effort QoS)
        cam01_color_sub = message_filters.Subscriber(self, Image, '/front_camera/color/image_raw/compressed', qos_profile=self.qos_profile)
        cam01_depth_sub = message_filters.Subscriber(self, Image, '/front_camera/depth/image_raw/compressedDepth', qos_profile=self.qos_profile)
        cam01_color_info_sub = message_filters.Subscriber(self, CameraInfo, '/front_camera/color/camera_info', qos_profile=self.qos_profile)
        cam01_depth_info_sub = message_filters.Subscriber(self, CameraInfo, '/front_camera/depth/camera_info', qos_profile=self.qos_profile)

        cam01_ats = message_filters.ApproximateTimeSynchronizer(
            [cam01_color_sub, cam01_depth_sub, cam01_color_info_sub, cam01_depth_info_sub], 
            queue_size=20, slop=0.05
        )
        cam01_ats.registerCallback(self.cam01_synced_callback)

        # Camera 02 subscribers (compressed topics, best_effort QoS)
        cam02_color_sub = message_filters.Subscriber(self, Image, '/top_camera/color/image_raw/compressed', qos_profile=self.qos_profile)
        cam02_depth_sub = message_filters.Subscriber(self, Image, '/top_camera/depth/image_raw/compressedDepth', qos_profile=self.qos_profile)
        cam02_color_info_sub = message_filters.Subscriber(self, CameraInfo, '/top_camera/color/camera_info', qos_profile=self.qos_profile)
        cam02_depth_info_sub = message_filters.Subscriber(self, CameraInfo, '/top_camera/depth/camera_info', qos_profile=self.qos_profile)

        cam02_ats = message_filters.ApproximateTimeSynchronizer(
            [cam02_color_sub, cam02_depth_sub, cam02_color_info_sub, cam02_depth_info_sub], 
            queue_size=20, slop=0.05
        )
        cam02_ats.registerCallback(self.cam02_synced_callback)

        self.get_logger().info('Unified Camera Node initialized (Front + Top cameras with color-depth overlay & best_effort QoS)')

    # ====================== GUI Setup ======================
    def setup_opencv_windows(self):
        try:
            cv2.namedWindow('Front Camera - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Front Camera - ROI Selection', self.mouse_callback_cam01)
            self.get_logger().info('OpenCV window setup for Front Camera ROI selection')
        except Exception as e:
            self.get_logger().error(f'Failed to setup Front Camera window: {str(e)}')

        try:
            cv2.namedWindow('Top Camera - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Top Camera - ROI Selection', self.mouse_callback_cam02)
            self.get_logger().info('OpenCV window setup for Top Camera ROI selection')
        except Exception as e:
            self.get_logger().error(f'Failed to setup Top Camera window: {str(e)}')

    def mouse_callback_cam01(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cam01_dragging = True
            self.cam01_start_point = (x, y)
            self.cam01_end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.cam01_dragging:
            self.cam01_end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.cam01_dragging and self.cam01_start_point:
                self.cam01_dragging = False
                self.cam01_end_point = (x, y)
                x1 = min(self.cam01_start_point[0], self.cam01_end_point[0])
                y1 = min(self.cam01_start_point[1], self.cam01_end_point[1])
                x2 = max(self.cam01_start_point[0], self.cam01_end_point[0])
                y2 = max(self.cam01_start_point[1], self.cam01_end_point[1])
                self.cam01_config['roi_x'] = x1
                self.cam01_config['roi_y'] = y1
                self.cam01_config['roi_width'] = x2 - x1
                self.cam01_config['roi_height'] = y2 - y1
                self.cam01_config['roi_enabled'] = True
                self.set_parameters([
                    Parameter('camera_01_roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('camera_01_roi_x', Parameter.Type.INTEGER, self.cam01_config['roi_x']),
                    Parameter('camera_01_roi_y', Parameter.Type.INTEGER, self.cam01_config['roi_y']),
                    Parameter('camera_01_roi_width', Parameter.Type.INTEGER, self.cam01_config['roi_width']),
                    Parameter('camera_01_roi_height', Parameter.Type.INTEGER, self.cam01_config['roi_height']),
                ])
                self.get_logger().info(f'Front Camera ROI set: x={self.cam01_config["roi_x"]}, y={self.cam01_config["roi_y"]}, w={self.cam01_config["roi_width"]}, h={self.cam01_config["roi_height"]}')

    def mouse_callback_cam02(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cam02_dragging = True
            self.cam02_start_point = (x, y)
            self.cam02_end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.cam02_dragging:
            self.cam02_end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.cam02_dragging and self.cam02_start_point:
                self.cam02_dragging = False
                self.cam02_end_point = (x, y)
                x1 = min(self.cam02_start_point[0], self.cam02_end_point[0])
                y1 = min(self.cam02_start_point[1], self.cam02_end_point[1])
                x2 = max(self.cam02_start_point[0], self.cam02_end_point[0])
                y2 = max(self.cam02_start_point[1], self.cam02_end_point[1])
                self.cam02_config['roi_x'] = x1
                self.cam02_config['roi_y'] = y1
                self.cam02_config['roi_width'] = x2 - x1
                self.cam02_config['roi_height'] = y2 - y1
                self.cam02_config['roi_enabled'] = True
                self.set_parameters([
                    Parameter('camera_02_roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('camera_02_roi_x', Parameter.Type.INTEGER, self.cam02_config['roi_x']),
                    Parameter('camera_02_roi_y', Parameter.Type.INTEGER, self.cam02_config['roi_y']),
                    Parameter('camera_02_roi_width', Parameter.Type.INTEGER, self.cam02_config['roi_width']),
                    Parameter('camera_02_roi_height', Parameter.Type.INTEGER, self.cam02_config['roi_height']),
                ])
                self.get_logger().info(f'Top Camera ROI set: x={self.cam02_config["roi_x"]}, y={self.cam02_config["roi_y"]}, w={self.cam02_config["roi_width"]}, h={self.cam02_config["roi_height"]}')

    # ====================== Camera 01 (Front) Callback ======================
    def cam01_synced_callback(self, color_msg: Image, depth_msg: Image, 
                              color_info: CameraInfo, depth_info: CameraInfo):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Front Camera color bridge error: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            # Normalize depth to meters
            if depth_msg.encoding == '16UC1':
                depth_m = depth.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_m = depth.astype(np.float32)
            else:
                depth_m = depth.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'Front Camera depth bridge error: {e}')
            return

        # Resize depth to match color dimensions
        if depth_m.shape != color.shape[:2]:
            depth_m = cv2.resize(depth_m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        annotated_image, pose_lm, face_lm, lhand_lm, rhand_lm, roi_ctx = self.process_camera_01(color)

        # Create overlay image
        overlay_image = self.overlay_depth_on_color(annotated_image, depth_m, self.cam01_config['overlay_alpha'])

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.cam01_annotated_pub.publish(ann)

        # Publish overlay image
        ovr = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
        ovr.header = color_msg.header
        self.cam01_overlay_pub.publish(ovr)

        # Publish landmarks
        self._publish_array(self.cam01_pose_landmarks_pub, pose_lm)
        self._publish_array(self.cam01_face_landmarks_pub, face_lm)
        self._publish_array(self.cam01_left_hand_landmarks_pub, lhand_lm)
        self._publish_array(self.cam01_right_hand_landmarks_pub, rhand_lm)

        # 3D projection & TF
        fx = depth_info.k[0]; fy = depth_info.k[4]
        cx = depth_info.k[2]; cy = depth_info.k[5]

        now = self.get_clock().now()
        if (now - self.cam01_last_tf_time).nanoseconds < (1e9 / self.cam01_config['tf_rate_hz']):
            return
        self.cam01_last_tf_time = now

        roi_x, roi_y, roi_w, roi_h, roi_enabled = roi_ctx

        def broadcast_set(flat_xyz, prefix):
            n = len(flat_xyz) // 3
            for i in range(n):
                u = float(flat_xyz[3*i + 0])
                v = float(flat_xyz[3*i + 1])
                u_i = int(np.clip(u, 0, depth_m.shape[1]-1))
                v_i = int(np.clip(v, 0, depth_m.shape[0]-1))
                z = self._robust_depth(depth_m, v_i, u_i)

                if not np.isfinite(z) or z <= 0.0:
                    continue

                X = (u - cx) / fx * z
                Y = (v - cy) / fy * z
                t = TransformStamped()
                t.header.stamp = color_msg.header.stamp
                t.header.frame_id = self.cam01_config['camera_frame']
                t.child_frame_id = f'{prefix}_{i}'
                t.transform.translation.x = float(X)
                t.transform.translation.y = float(Y)
                t.transform.translation.z = float(z)
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

        if pose_lm:
            broadcast_set(pose_lm, 'front_camera_pose')
        if lhand_lm:
            broadcast_set(lhand_lm, 'front_camera_left_hand')
        if rhand_lm:
            broadcast_set(rhand_lm, 'front_camera_right_hand')
        if self.cam01_config['publish_face_tf'] and face_lm:
            broadcast_set(face_lm, 'front_camera_face')

        # Display
        disp = overlay_image.copy()
        if self.cam01_config['roi_enabled'] and self.cam01_config['roi_width'] > 0 and self.cam01_config['roi_height'] > 0:
            cv2.rectangle(disp, (self.cam01_config['roi_x'], self.cam01_config['roi_y']),
                          (self.cam01_config['roi_x'] + self.cam01_config['roi_width'], self.cam01_config['roi_y'] + self.cam01_config['roi_height']), (0, 255, 0), 2)
            cv2.putText(disp, 'ROI', (self.cam01_config['roi_x'], self.cam01_config['roi_y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.cam01_dragging and self.cam01_start_point and self.cam01_end_point:
            cv2.rectangle(disp, self.cam01_start_point, self.cam01_end_point, (255, 0, 0), 2)
            cv2.putText(disp, 'Selecting ROI...',
                        (self.cam01_start_point[0], self.cam01_start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(disp, 'Drag to select ROI (q: quit, r: reset)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Front Camera - ROI Selection', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
        elif key == ord('r'):
            self.cam01_config['roi_enabled'] = False
            self.set_parameters([Parameter('camera_01_roi_enabled', Parameter.Type.BOOL, False)])
            self.get_logger().info('Front Camera ROI reset')

    # ====================== Camera 02 (Top) Callback ======================
    def cam02_synced_callback(self, color_msg: Image, depth_msg: Image, 
                              color_info: CameraInfo, depth_info: CameraInfo):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Top Camera color bridge error: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            # Normalize depth to meters
            if depth_msg.encoding == '16UC1':
                depth_m = depth.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_m = depth.astype(np.float32)
            else:
                depth_m = depth.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'Top Camera depth bridge error: {e}')
            return

        # Resize depth to match color dimensions
        if depth_m.shape != color.shape[:2]:
            depth_m = cv2.resize(depth_m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        annotated_image, lhand_lm, rhand_lm, roi_ctx = self.process_camera_02(color)

        # Create overlay image
        overlay_image = self.overlay_depth_on_color(annotated_image, depth_m, self.cam02_config['overlay_alpha'])

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.cam02_annotated_pub.publish(ann)

        # Publish overlay image
        ovr = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
        ovr.header = color_msg.header
        self.cam02_overlay_pub.publish(ovr)

        # Publish landmarks
        self._publish_array(self.cam02_left_hand_landmarks_pub, lhand_lm)
        self._publish_array(self.cam02_right_hand_landmarks_pub, rhand_lm)

        # 3D projection & TF
        fx = depth_info.k[0]; fy = depth_info.k[4]
        cx = depth_info.k[2]; cy = depth_info.k[5]

        now = self.get_clock().now()
        if (now - self.cam02_last_tf_time).nanoseconds < (1e9 / self.cam02_config['tf_rate_hz']):
            return
        self.cam02_last_tf_time = now

        roi_x, roi_y, roi_w, roi_h, roi_enabled = roi_ctx

        def broadcast_set(flat_xyz, prefix):
            n = len(flat_xyz) // 3
            for i in range(n):
                u = float(flat_xyz[3*i + 0])
                v = float(flat_xyz[3*i + 1])
                u_i = int(np.clip(u, 0, depth_m.shape[1]-1))
                v_i = int(np.clip(v, 0, depth_m.shape[0]-1))
                z = self._robust_depth(depth_m, v_i, u_i)

                if not np.isfinite(z) or z <= 0.0:
                    continue

                X = (u - cx) / fx * z
                Y = (v - cy) / fy * z
                t = TransformStamped()
                t.header.stamp = color_msg.header.stamp
                t.header.frame_id = self.cam02_config['camera_frame']
                t.child_frame_id = f'{prefix}_{i}'
                t.transform.translation.x = float(X)
                t.transform.translation.y = float(Y)
                t.transform.translation.z = float(z)
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

        if lhand_lm:
            broadcast_set(lhand_lm, 'top_camera_left_hand')
        if rhand_lm:
            broadcast_set(rhand_lm, 'top_camera_right_hand')

        # Display
        disp = overlay_image.copy()
        if self.cam02_config['roi_enabled'] and self.cam02_config['roi_width'] > 0 and self.cam02_config['roi_height'] > 0:
            cv2.rectangle(disp, (self.cam02_config['roi_x'], self.cam02_config['roi_y']),
                          (self.cam02_config['roi_x'] + self.cam02_config['roi_width'], self.cam02_config['roi_y'] + self.cam02_config['roi_height']), (0, 255, 0), 2)
            cv2.putText(disp, 'ROI', (self.cam02_config['roi_x'], self.cam02_config['roi_y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.cam02_dragging and self.cam02_start_point and self.cam02_end_point:
            cv2.rectangle(disp, self.cam02_start_point, self.cam02_end_point, (255, 0, 0), 2)
            cv2.putText(disp, 'Selecting ROI...',
                        (self.cam02_start_point[0], self.cam02_start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(disp, 'Drag to select ROI (q: quit, r: reset)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Top Camera - ROI Selection', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
        elif key == ord('r'):
            self.cam02_config['roi_enabled'] = False
            self.set_parameters([Parameter('camera_02_roi_enabled', Parameter.Type.BOOL, False)])
            self.get_logger().info('Top Camera ROI reset')

    # ====================== Processing Methods ======================
    def process_camera_01(self, cv_image):
        """Process front camera: Holistic + Face Mesh"""
        height, width = cv_image.shape[:2]

        # ROI cropping
        if self.cam01_config['roi_enabled'] and self.cam01_config['roi_width'] > 0 and self.cam01_config['roi_height'] > 0:
            roi_x = int(np.clip(self.cam01_config['roi_x'], 0, width-1))
            roi_y = int(np.clip(self.cam01_config['roi_y'], 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.cam01_config['roi_width'], 0, width))
            roi_y2 = int(np.clip(roi_y + self.cam01_config['roi_height'], 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height

        # MediaPipe processing
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        holistic_results = self.holistic.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if holistic_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        if holistic_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

        if holistic_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Stitch back
        if self.cam01_config['roi_enabled'] and self.cam01_config['roi_width'] > 0 and self.cam01_config['roi_height'] > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        # Extract landmarks
        pose_landmarks = self.extract_pose_landmarks(holistic_results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        face_landmarks = self.extract_face_landmarks(face_results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        left_hand_landmarks = self.extract_hand_landmarks(holistic_results.left_hand_landmarks, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        right_hand_landmarks = self.extract_hand_landmarks(holistic_results.right_hand_landmarks, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))

        roi_ctx = (self.cam01_config['roi_x'], self.cam01_config['roi_y'], self.cam01_config['roi_width'], self.cam01_config['roi_height'], self.cam01_config['roi_enabled'])
        return full_annotated, pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks, roi_ctx

    def process_camera_02(self, cv_image):
        """Process top camera: Hands only"""
        height, width = cv_image.shape[:2]

        # ROI cropping
        if self.cam02_config['roi_enabled'] and self.cam02_config['roi_width'] > 0 and self.cam02_config['roi_height'] > 0:
            roi_x = int(np.clip(self.cam02_config['roi_x'], 0, width-1))
            roi_y = int(np.clip(self.cam02_config['roi_y'], 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.cam02_config['roi_width'], 0, width))
            roi_y2 = int(np.clip(roi_y + self.cam02_config['roi_height'], 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height

        # MediaPipe processing
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        hands_results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        left_hand_landmarks = []
        right_hand_landmarks = []

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label
                self.mp_drawing.draw_landmarks(
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
                flat = self.extract_hand_landmarks(hand_landmarks, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
                if handedness == "Left":
                    left_hand_landmarks = flat
                elif handedness == "Right":
                    right_hand_landmarks = flat

        # Stitch back
        if self.cam02_config['roi_enabled'] and self.cam02_config['roi_width'] > 0 and self.cam02_config['roi_height'] > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        roi_ctx = (self.cam02_config['roi_x'], self.cam02_config['roi_y'], self.cam02_config['roi_width'], self.cam02_config['roi_height'], self.cam02_config['roi_enabled'])
        return full_annotated, left_hand_landmarks, right_hand_landmarks, roi_ctx

    # ====================== Helper Methods ======================
    def _publish_array(self, pub, flat):
        msg = Float32MultiArray()
        msg.data = flat
        pub.publish(msg)

    def _robust_depth(self, depth_m, v, u):
        """Median depth from 3x3 patch, ignoring zeros"""
        h, w = depth_m.shape
        v0 = max(0, v-1); v1 = min(h, v+2)
        u0 = max(0, u-1); u1 = min(w, u+2)
        patch = depth_m[v0:v1, u0:u1].reshape(-1)
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def overlay_depth_on_color(self, color, depth_m, alpha=0.3):
        """Overlay depth visualization on color image"""
        # Normalize depth to 0-255 range (adjust factor as needed)
        depth_normalized = np.clip((depth_m / 3.0) * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap to depth
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Blend color and depth visualization
        overlaid = cv2.addWeighted(color, 1.0 - alpha, depth_color, alpha, 0)
        
        return overlaid

    def extract_pose_landmarks(self, results, width, height, roi_offset, roi_bbox):
        landmarks = []
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
                z = lm.z
                landmarks.extend([x, y, z])
        return landmarks

    def extract_face_landmarks(self, results, width, height, roi_offset, roi_bbox):
        landmarks = []
        if results and results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                for lm in face_lm.landmark:
                    x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                    y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
                    z = lm.z
                    landmarks.extend([x, y, z])
        return landmarks

    def extract_hand_landmarks(self, hand_lm, width, height, roi_offset, roi_bbox):
        landmarks = []
        if hand_lm:
            for lm in hand_lm.landmark:
                x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
                z = lm.z
                landmarks.extend([x, y, z])
        return landmarks


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedCameraNode()
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