# unified_camera_node.py
# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class UnifiedCameraNode(Node):
    def __init__(self):
        super().__init__('unified_camera')

        # CvBridgeのインスタンスを作成
        self.bridge = CvBridge()

        # ==== MediaPipe初期化 ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        # Holistic (姿勢 + 手)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Hands (Top Camera用)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # QoS設定
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==== Subscribers ====
        # FRONTカメラ
        front_color_sub = Subscriber(self, Image, '/front_camera/color/image_raw', qos_profile=qos_profile)
        front_depth_sub = Subscriber(self, Image, '/front_camera/depth/image_raw', qos_profile=qos_profile)
        
        # TOPカメラ
        top_color_sub = Subscriber(self, Image, '/top_camera/color/image_raw', qos_profile=qos_profile)
        top_depth_sub = Subscriber(self, Image, '/top_camera/depth/image_raw', qos_profile=qos_profile)

        # メッセージ同期
        self.camera_sync = ApproximateTimeSynchronizer(
            [front_color_sub, front_depth_sub, top_color_sub, top_depth_sub],
            queue_size=50,
            slop=0.5
        )
        self.camera_sync.registerCallback(self.camera_callback)

        # ==== Publishers ====
        self.front_annotated_pub = self.create_publisher(Image, '/front_camera/annotated_image', 10)
        self.front_overlay_pub = self.create_publisher(Image, '/front_camera/overlay_image', 10)
        self.top_annotated_pub = self.create_publisher(Image, '/top_camera/annotated_image', 10)
        self.top_overlay_pub = self.create_publisher(Image, '/top_camera/overlay_image', 10)

        # ランドマーク出力
        self.front_pose_pub = self.create_publisher(Float32MultiArray, '/front_camera/pose_landmarks', 10)
        self.front_face_pub = self.create_publisher(Float32MultiArray, '/front_camera/face_landmarks', 10)
        self.top_hand_left_pub = self.create_publisher(Float32MultiArray, '/top_camera/left_hand_landmarks', 10)
        self.top_hand_right_pub = self.create_publisher(Float32MultiArray, '/top_camera/right_hand_landmarks', 10)

        # ウィンドウ作成
        cv2.namedWindow("FRONT Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FRONT Camera", 640, 480)
        cv2.resizeWindow("TOP Camera", 640, 480)

        self.get_logger().info('Unified Camera Node initialized')

    def camera_callback(self, front_color_msg, front_depth_msg, top_color_msg, top_depth_msg):
        try:
            # ==== FRONT Camera処理 ====
            front_color = self.bridge.imgmsg_to_cv2(front_color_msg, desired_encoding='bgr8')
            front_depth = self.bridge.imgmsg_to_cv2(front_depth_msg, desired_encoding='passthrough')

            # FRONT: Holistic + Face Mesh処理
            front_annotated, pose_lm, face_lm = self.process_front_camera(front_color)
            
            # 深度の可視化
            front_depth_normalized = cv2.normalize(front_depth, None, 0, 255, cv2.NORM_MINMAX)
            front_depth_8bit = cv2.convertScaleAbs(front_depth_normalized)
            front_depth_colored = cv2.applyColorMap(front_depth_8bit, cv2.COLORMAP_JET)

            # オーバーレイ合成
            front_overlay = cv2.addWeighted(front_annotated, 0.7, front_depth_colored, 0.3, 0)

            # 出力
            self.publish_image(self.front_annotated_pub, front_annotated, front_color_msg)
            self.publish_image(self.front_overlay_pub, front_overlay, front_color_msg)
            self.publish_landmarks(self.front_pose_pub, pose_lm)
            self.publish_landmarks(self.front_face_pub, face_lm)

            # ==== TOP Camera処理 ====
            top_color = self.bridge.imgmsg_to_cv2(top_color_msg, desired_encoding='bgr8')
            top_depth = self.bridge.imgmsg_to_cv2(top_depth_msg, desired_encoding='passthrough')

            # TOP: 手検出のみ
            top_annotated, left_hand, right_hand = self.process_top_camera(top_color)

            # 深度の可視化
            top_depth_normalized = cv2.normalize(top_depth, None, 0, 255, cv2.NORM_MINMAX)
            top_depth_8bit = cv2.convertScaleAbs(top_depth_normalized)
            top_depth_colored = cv2.applyColorMap(top_depth_8bit, cv2.COLORMAP_JET)

            # オーバーレイ合成
            top_overlay = cv2.addWeighted(top_annotated, 0.7, top_depth_colored, 0.3, 0)

            # 出力
            self.publish_image(self.top_annotated_pub, top_annotated, top_color_msg)
            self.publish_image(self.top_overlay_pub, top_overlay, top_color_msg)
            self.publish_landmarks(self.top_hand_left_pub, left_hand)
            self.publish_landmarks(self.top_hand_right_pub, right_hand)

            # ==== Display ====
            cv2.imshow("FRONT Camera", front_overlay)
            cv2.imshow("TOP Camera", top_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()

        except Exception as e:
            self.get_logger().error(f"Error processing camera images: {e}")

    def process_front_camera(self, cv_image):
        """Front Camera: Holistic + Face Mesh"""
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Holistic処理
        holistic_results = self.holistic.process(image_rgb)
        
        # Face Mesh処理
        face_results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # ランドマーク描画
        if holistic_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        if holistic_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        if holistic_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, holistic_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)

        # ランドマーク抽出
        pose_lm = self.extract_landmarks(holistic_results.pose_landmarks)
        face_lm = self.extract_landmarks_from_multi(face_results.multi_face_landmarks)

        return annotated, pose_lm, face_lm

    def process_top_camera(self, cv_image):
        """Top Camera: Hands only"""
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        hands_results = self.hands.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        left_hand = []
        right_hand = []

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label
                
                self.mp_drawing.draw_landmarks(
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                hand_lm = self.extract_landmarks(hand_landmarks)
                
                if handedness == "Left":
                    left_hand = hand_lm
                else:
                    right_hand = hand_lm

        return annotated, left_hand, right_hand

    def extract_landmarks(self, landmarks):
        """ランドマークをフラット配列に変換"""
        flat = []
        if landmarks:
            for lm in landmarks.landmark:
                flat.extend([lm.x, lm.y, lm.z])
        return flat

    def extract_landmarks_from_multi(self, multi_landmarks):
        """複数のランドマークをフラット配列に変換"""
        flat = []
        if multi_landmarks:
            for landmarks in multi_landmarks:
                for lm in landmarks.landmark:
                    flat.extend([lm.x, lm.y, lm.z])
        return flat

    def publish_image(self, pub, cv_image, header_msg):
        """画像をパブリッシュ"""
        msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        msg.header = header_msg.header
        pub.publish(msg)

    def publish_landmarks(self, pub, landmarks):
        """ランドマークをパブリッシュ"""
        msg = Float32MultiArray()
        msg.data = landmarks
        pub.publish(msg)


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