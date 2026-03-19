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

        # ==== Parameters ====
        self.declare_parameter('camera_01_min_detection_confidence', 0.6)
        self.declare_parameter('camera_01_min_tracking_confidence', 0.6)
        self.declare_parameter('camera_01_camera_frame', 'camera_01_depth_optical_frame')
        self.declare_parameter('camera_02_min_detection_confidence', 0.6)
        self.declare_parameter('camera_02_min_tracking_confidence', 0.6)
        self.declare_parameter('camera_02_camera_frame', 'camera_02_depth_optical_frame')

        # Read parameters
        self.cam01_config = {
            'min_detection_confidence': float(self.get_parameter('camera_01_min_detection_confidence').value),
            'min_tracking_confidence': float(self.get_parameter('camera_01_min_tracking_confidence').value),
            'camera_frame': self.get_parameter('camera_01_camera_frame').value,
        }

        self.cam02_config = {
            'min_detection_confidence': float(self.get_parameter('camera_02_min_detection_confidence').value),
            'min_tracking_confidence': float(self.get_parameter('camera_02_min_tracking_confidence').value),
            'camera_frame': self.get_parameter('camera_02_camera_frame').value,
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

        # ==== GUI State ====
        self.cam01_window_created = False
        self.cam02_window_created = False

        # ==== Latest CameraInfo ====
        self.cam01_color_info = None
        self.cam01_depth_info = None
        self.cam02_color_info = None
        self.cam02_depth_info = None

        # ==== Publishers ====
        self.cam01_annotated_pub = self.create_publisher(Image, '/front_camera/annotated_image', self.qos_profile)
        self.cam01_overlay_pub = self.create_publisher(Image, '/front_camera/overlay_image', self.qos_profile)
        self.cam02_annotated_pub = self.create_publisher(Image, '/top_camera/annotated_image', self.qos_profile)
        self.cam02_overlay_pub = self.create_publisher(Image, '/top_camera/overlay_image', self.qos_profile)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)

        # ==== Subscribers with message synchronization (color + depth only) ====
        self.get_logger().info('Setting up Front Camera synchronizer...')
        cam01_color_sub = message_filters.Subscriber(self, Image, '/front_camera/color/image_raw/compressed', qos_profile=self.qos_profile)
        cam01_depth_sub = message_filters.Subscriber(self, Image, '/front_camera/depth/image_raw/compressed', qos_profile=self.qos_profile)

        cam01_ats = message_filters.ApproximateTimeSynchronizer(
            [cam01_color_sub, cam01_depth_sub], 
            queue_size=30, slop=0.1
        )
        cam01_ats.registerCallback(self.cam01_synced_callback)
        self.get_logger().info('Front Camera synchronizer registered')

        # CameraInfo subscribers (separate, no time sync)
        self.create_subscription(CameraInfo, '/front_camera/color/camera_info', self.cam01_color_info_callback, self.qos_profile)
        self.create_subscription(CameraInfo, '/front_camera/depth/camera_info', self.cam01_depth_info_callback, self.qos_profile)

        self.get_logger().info('Setting up Top Camera synchronizer...')
        cam02_color_sub = message_filters.Subscriber(self, Image, '/top_camera/color/image_raw/compressed', qos_profile=self.qos_profile)
        cam02_depth_sub = message_filters.Subscriber(self, Image, '/top_camera/depth/image_raw/compressed', qos_profile=self.qos_profile)

        cam02_ats = message_filters.ApproximateTimeSynchronizer(
            [cam02_color_sub, cam02_depth_sub], 
            queue_size=30, slop=0.1
        )
        cam02_ats.registerCallback(self.cam02_synced_callback)
        self.get_logger().info('Top Camera synchronizer registered')

        # CameraInfo subscribers (separate, no time sync)
        self.create_subscription(CameraInfo, '/top_camera/color/camera_info', self.cam02_color_info_callback, self.qos_profile)
        self.create_subscription(CameraInfo, '/top_camera/depth/camera_info', self.cam02_depth_info_callback, self.qos_profile)

        self.get_logger().info('Unified Camera Node initialized successfully')

    # ====================== CameraInfo Callbacks ======================
    def cam01_color_info_callback(self, msg: CameraInfo):
        self.cam01_color_info = msg

    def cam01_depth_info_callback(self, msg: CameraInfo):
        self.cam01_depth_info = msg

    def cam02_color_info_callback(self, msg: CameraInfo):
        self.cam02_color_info = msg

    def cam02_depth_info_callback(self, msg: CameraInfo):
        self.cam02_depth_info = msg

    # ====================== Camera 01 (Front) Callback ======================
    def cam01_synced_callback(self, color_msg: Image, depth_msg: Image):
        self.get_logger().info('Front Camera callback received')
        
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.get_logger().info(f'✓ Front Camera color: {color.shape}')
        except Exception as e:
            self.get_logger().error(f'✗ Front Camera color error: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            self.get_logger().info(f'✓ Front Camera depth: {depth_m.shape}')
        except Exception as e:
            self.get_logger().error(f'✗ Front Camera depth error: {e}')
            return

        # Resize depth to match color
        if depth_m.shape != color.shape[:2]:
            depth_m = cv2.resize(depth_m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Process
        annotated_image, pose_lm, face_lm, lhand_lm, rhand_lm = self.process_camera_01(color)
        overlay_image = self.overlay_depth_on_color(annotated_image, depth_m, 0.3)

        # Publish
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.cam01_annotated_pub.publish(ann)

        ovr = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
        ovr.header = color_msg.header
        self.cam01_overlay_pub.publish(ovr)

        # Display
        if not self.cam01_window_created:
            cv2.namedWindow('Front Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Front Camera', 640, 480)
            self.cam01_window_created = True

        disp = overlay_image.copy()
        cv2.putText(disp, 'Front Camera (q: quit)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Front Camera', disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()

    # ====================== Camera 02 (Top) Callback ======================
    def cam02_synced_callback(self, color_msg: Image, depth_msg: Image):
        self.get_logger().info('Top Camera callback received')
        
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.get_logger().info(f'✓ Top Camera color: {color.shape}')
        except Exception as e:
            self.get_logger().error(f'✗ Top Camera color error: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            self.get_logger().info(f'✓ Top Camera depth: {depth_m.shape}')
        except Exception as e:
            self.get_logger().error(f'✗ Top Camera depth error: {e}')
            return

        # Resize depth to match color
        if depth_m.shape != color.shape[:2]:
            depth_m = cv2.resize(depth_m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Process
        annotated_image, lhand_lm, rhand_lm = self.process_camera_02(color)
        overlay_image = self.overlay_depth_on_color(annotated_image, depth_m, 0.3)

        # Publish
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.cam02_annotated_pub.publish(ann)

        ovr = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
        ovr.header = color_msg.header
        self.cam02_overlay_pub.publish(ovr)

        # Display
        if not self.cam02_window_created:
            cv2.namedWindow('Top Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Top Camera', 640, 480)
            self.cam02_window_created = True

        disp = overlay_image.copy()
        cv2.putText(disp, 'Top Camera (q: quit)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Top Camera', disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()

    # ====================== Processing Methods ======================
    def process_camera_01(self, cv_image):
        """Process front camera"""
        height, width = cv_image.shape[:2]
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        holistic_results = self.holistic.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if holistic_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(annotated, holistic_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if holistic_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(annotated, holistic_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if holistic_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(annotated, holistic_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(annotated, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)

        pose_lm = self.extract_pose_landmarks(holistic_results)
        face_lm = self.extract_face_landmarks(face_results)
        lhand_lm = self.extract_hand_landmarks(holistic_results.left_hand_landmarks)
        rhand_lm = self.extract_hand_landmarks(holistic_results.right_hand_landmarks)

        return annotated, pose_lm, face_lm, lhand_lm, rhand_lm

    def process_camera_02(self, cv_image):
        """Process top camera"""
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        hands_results = self.hands.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        lhand_lm = []
        rhand_lm = []

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label
                self.mp_drawing.draw_landmarks(annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                flat = self.extract_hand_landmarks(hand_landmarks)
                if handedness == "Left":
                    lhand_lm = flat
                elif handedness == "Right":
                    rhand_lm = flat

        return annotated, lhand_lm, rhand_lm

    # ====================== Helper Methods ======================
    def overlay_depth_on_color(self, color, depth_m, alpha=0.3):
        """Overlay depth visualization on color image"""
        if color is None or depth_m is None:
            return color if color is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        depth_normalized = np.clip((depth_m / 3.0) * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        overlaid = cv2.addWeighted(color, 1.0 - alpha, depth_color, alpha, 0)
        return overlaid

    def extract_pose_landmarks(self, results):
        landmarks = []
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

    def extract_face_landmarks(self, results):
        landmarks = []
        if results and results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                for lm in face_lm.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

    def extract_hand_landmarks(self, hand_lm):
        landmarks = []
        if hand_lm:
            for lm in hand_lm.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
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