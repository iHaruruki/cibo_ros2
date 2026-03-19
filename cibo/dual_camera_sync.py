# unified_camera_node.py
# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
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

        self.get_logger().info("Setting up camera subscribers...")
        
        # ==== Subscribers ====
        # FRONTカメラ：圧縮カラー + 非圧縮深度
        front_color_sub = Subscriber(self, CompressedImage, '/front_camera/color/image_raw/compressed', qos_profile=qos_profile)
        front_depth_sub = Subscriber(self, Image, '/front_camera/depth/image_raw', qos_profile=qos_profile)
        
        # TOPカメラ：圧縮カラー + 非圧縮深度
        top_color_sub = Subscriber(self, CompressedImage, '/top_camera/color/image_raw/compressed', qos_profile=qos_profile)
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
        self.top_annotated_pub = self.create_publisher(Image, '/top_camera/annotated_image', 10)

        # ランドマーク出力
        self.front_pose_pub = self.create_publisher(Float32MultiArray, '/front_camera/pose_landmarks', 10)
        self.front_face_pub = self.create_publisher(Float32MultiArray, '/front_camera/face_landmarks', 10)
        self.front_left_hand_pub = self.create_publisher(Float32MultiArray, '/front_camera/left_hand_landmarks', 10)
        self.front_right_hand_pub = self.create_publisher(Float32MultiArray, '/front_camera/right_hand_landmarks', 10)
        self.top_left_hand_pub = self.create_publisher(Float32MultiArray, '/top_camera/left_hand_landmarks', 10)
        self.top_right_hand_pub = self.create_publisher(Float32MultiArray, '/top_camera/right_hand_landmarks', 10)

        # ウィンドウ作成
        cv2.namedWindow("FRONT Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("TOP Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FRONT Camera", 640, 480)
        cv2.resizeWindow("TOP Camera", 640, 480)

        self.frame_count = 0
        self.get_logger().info('Unified Camera Node initialized')

    def decompress_color_image(self, compressed_msg):
        """圧縮カラー画像をデコードする"""
        try:
            if len(compressed_msg.data) == 0:
                return None
            compressed_data = np.frombuffer(compressed_msg.data, np.uint8)
            image = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            self.get_logger().error(f"Error decompressing color image: {e}")
            return None

    def camera_callback(self, front_color_msg, front_depth_msg, top_color_msg, top_depth_msg):
        try:
            self.frame_count += 1

            # ==== FRONT Camera処理 ====
            front_color = self.decompress_color_image(front_color_msg)
            if front_color is None:
                return

            # Holistic + Face Mesh処理
            front_annotated, pose_lm, face_lm, front_left_hand, front_right_hand = self.process_front_camera(front_color)
            
            # 出力
            self.publish_image(self.front_annotated_pub, front_annotated, front_color_msg)
            self.publish_landmarks(self.front_pose_pub, pose_lm)
            self.publish_landmarks(self.front_face_pub, face_lm)
            self.publish_landmarks(self.front_left_hand_pub, front_left_hand)
            self.publish_landmarks(self.front_right_hand_pub, front_right_hand)

            # ==== TOP Camera処理 ====
            top_color = self.decompress_color_image(top_color_msg)
            if top_color is None:
                return

            # 手検出のみ
            top_annotated, top_left_hand, top_right_hand = self.process_top_camera(top_color)

            # 出力
            self.publish_image(self.top_annotated_pub, top_annotated, top_color_msg)
            self.publish_landmarks(self.top_left_hand_pub, top_left_hand)
            self.publish_landmarks(self.top_right_hand_pub, top_right_hand)

            # ==== Display ====
            cv2.imshow("FRONT Camera", front_annotated)
            cv2.imshow("TOP Camera", top_annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()

            if self.frame_count % 60 == 0:
                self.get_logger().info(f"Processed {self.frame_count} frames")

        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {e}")

    def process_front_camera(self, cv_image):
        """Front Camera: Holistic + Face Mesh"""
        height, width = cv_image.shape[:2]
        
        # BGR→RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        holistic_results = self.holistic.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw holistic
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

        # Face mesh (with iris)
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

        # Extract landmarks
        pose_lm = self.extract_pose_landmarks(holistic_results, width, height)
        face_lm = self.extract_face_landmarks(face_results, width, height)
        left_hand = self.extract_hand_landmarks(holistic_results.left_hand_landmarks, width, height)
        right_hand = self.extract_hand_landmarks(holistic_results.right_hand_landmarks, width, height)

        return annotated, pose_lm, face_lm, left_hand, right_hand

    def process_top_camera(self, cv_image):
        """Top Camera: Hands only"""
        height, width = cv_image.shape[:2]
        
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
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
                
                hand_lm = self.extract_hand_landmarks(hand_landmarks, width, height)
                
                if handedness == "Left":
                    left_hand = hand_lm
                else:
                    right_hand = hand_lm

        return annotated, left_hand, right_hand

    def extract_pose_landmarks(self, results, width, height):
        """ランドマークを抽出（ピクセル座標 + MediaPipe Z）"""
        landmarks = []
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = lm.x * width
                y = lm.y * height
                z = lm.z
                landmarks.extend([x, y, z])
        return landmarks

    def extract_face_landmarks(self, results, width, height):
        """顔ランドマークを抽出"""
        landmarks = []
        if results and results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                for lm in face_lm.landmark:
                    x = lm.x * width
                    y = lm.y * height
                    z = lm.z
                    landmarks.extend([x, y, z])
        return landmarks

    def extract_hand_landmarks(self, hand_lm, width, height):
        """手ランドマークを抽出"""
        landmarks = []
        if hand_lm:
            for lm in hand_lm.landmark:
                x = lm.x * width
                y = lm.y * height
                z = lm.z
                landmarks.extend([x, y, z])
        return landmarks

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