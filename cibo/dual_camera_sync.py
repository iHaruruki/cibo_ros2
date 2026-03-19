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
from rclpy.parameter import Parameter

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

        # ==== Parameters (ROI設定) ====
        self.declare_parameter('front_roi_enabled', False)
        self.declare_parameter('front_roi_x', 0)
        self.declare_parameter('front_roi_y', 0)
        self.declare_parameter('front_roi_width', 400)
        self.declare_parameter('front_roi_height', 300)
        
        self.declare_parameter('top_roi_enabled', False)
        self.declare_parameter('top_roi_x', 0)
        self.declare_parameter('top_roi_y', 0)
        self.declare_parameter('top_roi_width', 400)
        self.declare_parameter('top_roi_height', 300)

        # ROI設定を読み込む
        self.front_roi_enabled = bool(self.get_parameter('front_roi_enabled').value)
        self.front_roi_x = int(self.get_parameter('front_roi_x').value)
        self.front_roi_y = int(self.get_parameter('front_roi_y').value)
        self.front_roi_width = int(self.get_parameter('front_roi_width').value)
        self.front_roi_height = int(self.get_parameter('front_roi_height').value)
        
        self.top_roi_enabled = bool(self.get_parameter('top_roi_enabled').value)
        self.top_roi_x = int(self.get_parameter('top_roi_x').value)
        self.top_roi_y = int(self.get_parameter('top_roi_y').value)
        self.top_roi_width = int(self.get_parameter('top_roi_width').value)
        self.top_roi_height = int(self.get_parameter('top_roi_height').value)

        # ==== ROI GUI State ====
        self.front_dragging = False
        self.front_start_point = None
        self.front_end_point = None
        
        self.top_dragging = False
        self.top_start_point = None
        self.top_end_point = None

        self.setup_opencv_windows()

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

        self.frame_count = 0
        self.get_logger().info('Unified Camera Node initialized with ROI support')

    def setup_opencv_windows(self):
        """OpenCVウィンドウのセットアップ"""
        try:
            cv2.namedWindow("FRONT Camera - ROI Selection", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("FRONT Camera - ROI Selection", self.mouse_callback_front)
            cv2.resizeWindow("FRONT Camera - ROI Selection", 640, 480)
            self.get_logger().info('Front Camera ROI window setup')
        except Exception as e:
            self.get_logger().error(f'Failed to setup Front Camera window: {str(e)}')

        try:
            cv2.namedWindow("TOP Camera - ROI Selection", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("TOP Camera - ROI Selection", self.mouse_callback_top)
            cv2.resizeWindow("TOP Camera - ROI Selection", 640, 480)
            self.get_logger().info('Top Camera ROI window setup')
        except Exception as e:
            self.get_logger().error(f'Failed to setup Top Camera window: {str(e)}')

    def mouse_callback_front(self, event, x, y, flags, param):
        """Front Camera ROIマウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.front_dragging = True
            self.front_start_point = (x, y)
            self.front_end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.front_dragging:
            self.front_end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.front_dragging and self.front_start_point:
                self.front_dragging = False
                self.front_end_point = (x, y)
                x1 = min(self.front_start_point[0], self.front_end_point[0])
                y1 = min(self.front_start_point[1], self.front_end_point[1])
                x2 = max(self.front_start_point[0], self.front_end_point[0])
                y2 = max(self.front_start_point[1], self.front_end_point[1])
                self.front_roi_x = x1
                self.front_roi_y = y1
                self.front_roi_width = x2 - x1
                self.front_roi_height = y2 - y1
                self.front_roi_enabled = True
                self.set_parameters([
                    Parameter('front_roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('front_roi_x', Parameter.Type.INTEGER, self.front_roi_x),
                    Parameter('front_roi_y', Parameter.Type.INTEGER, self.front_roi_y),
                    Parameter('front_roi_width', Parameter.Type.INTEGER, self.front_roi_width),
                    Parameter('front_roi_height', Parameter.Type.INTEGER, self.front_roi_height),
                ])
                self.get_logger().info(f'Front ROI set: x={self.front_roi_x}, y={self.front_roi_y}, w={self.front_roi_width}, h={self.front_roi_height}')

    def mouse_callback_top(self, event, x, y, flags, param):
        """Top Camera ROIマウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.top_dragging = True
            self.top_start_point = (x, y)
            self.top_end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.top_dragging:
            self.top_end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.top_dragging and self.top_start_point:
                self.top_dragging = False
                self.top_end_point = (x, y)
                x1 = min(self.top_start_point[0], self.top_end_point[0])
                y1 = min(self.top_start_point[1], self.top_end_point[1])
                x2 = max(self.top_start_point[0], self.top_end_point[0])
                y2 = max(self.top_start_point[1], self.top_end_point[1])
                self.top_roi_x = x1
                self.top_roi_y = y1
                self.top_roi_width = x2 - x1
                self.top_roi_height = y2 - y1
                self.top_roi_enabled = True
                self.set_parameters([
                    Parameter('top_roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('top_roi_x', Parameter.Type.INTEGER, self.top_roi_x),
                    Parameter('top_roi_y', Parameter.Type.INTEGER, self.top_roi_y),
                    Parameter('top_roi_width', Parameter.Type.INTEGER, self.top_roi_width),
                    Parameter('top_roi_height', Parameter.Type.INTEGER, self.top_roi_height),
                ])
                self.get_logger().info(f'Top ROI set: x={self.top_roi_x}, y={self.top_roi_y}, w={self.top_roi_width}, h={self.top_roi_height}')

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

            # ==== Display with ROI ====
            self.display_with_roi(front_annotated, "FRONT Camera - ROI Selection", 
                                  self.front_roi_enabled, self.front_roi_x, self.front_roi_y, 
                                  self.front_roi_width, self.front_roi_height,
                                  self.front_dragging, self.front_start_point, self.front_end_point)
            
            self.display_with_roi(top_annotated, "TOP Camera - ROI Selection",
                                  self.top_roi_enabled, self.top_roi_x, self.top_roi_y,
                                  self.top_roi_width, self.top_roi_height,
                                  self.top_dragging, self.top_start_point, self.top_end_point)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
            elif key == ord('r'):
                # Front ROI リセット
                self.front_roi_enabled = False
                self.set_parameters([Parameter('front_roi_enabled', Parameter.Type.BOOL, False)])
                self.get_logger().info('Front ROI reset')
            elif key == ord('t'):
                # Top ROI リセット
                self.top_roi_enabled = False
                self.set_parameters([Parameter('top_roi_enabled', Parameter.Type.BOOL, False)])
                self.get_logger().info('Top ROI reset')

            if self.frame_count % 60 == 0:
                self.get_logger().info(f"Processed {self.frame_count} frames")

        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {e}")

    def display_with_roi(self, image, window_name, roi_enabled, roi_x, roi_y, roi_width, roi_height,
                         dragging, start_point, end_point):
        """ROI表示付きで画像を表示"""
        disp = image.copy()
        
        # ROIが有効な場合、矩形を描画
        if roi_enabled and roi_width > 0 and roi_height > 0:
            cv2.rectangle(disp, (roi_x, roi_y),
                          (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
            cv2.putText(disp, 'ROI', (roi_x, roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ドラッグ中の矩形を描画
        if dragging and start_point and end_point:
            cv2.rectangle(disp, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(disp, 'Selecting ROI...',
                        (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # ガイドテキスト
        cv2.putText(disp, 'Drag to select ROI (q: quit, r/t: reset)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, disp)

    def process_front_camera(self, cv_image):
        """Front Camera: Holistic + Face Mesh with ROI support"""
        height, width = cv_image.shape[:2]
        
        # ROI cropping
        if self.front_roi_enabled and self.front_roi_width > 0 and self.front_roi_height > 0:
            roi_x = int(np.clip(self.front_roi_x, 0, width-1))
            roi_y = int(np.clip(self.front_roi_y, 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.front_roi_width, 0, width))
            roi_y2 = int(np.clip(roi_y + self.front_roi_height, 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
            roi_bbox = (roi_x, roi_y, roi_x2, roi_y2)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height
            roi_bbox = (0, 0, width, height)
        
        # BGR→RGB
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
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

        # Stitch back into full image if ROI
        if self.front_roi_enabled and self.front_roi_width > 0 and self.front_roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        # Extract landmarks
        pose_lm = self.extract_pose_landmarks(holistic_results, width, height, roi_offset, roi_bbox)
        face_lm = self.extract_face_landmarks(face_results, width, height, roi_offset, roi_bbox)
        left_hand = self.extract_hand_landmarks(holistic_results.left_hand_landmarks, width, height, roi_offset, roi_bbox)
        right_hand = self.extract_hand_landmarks(holistic_results.right_hand_landmarks, width, height, roi_offset, roi_bbox)

        return full_annotated, pose_lm, face_lm, left_hand, right_hand

    def process_top_camera(self, cv_image):
        """Top Camera: Hands only with ROI support"""
        height, width = cv_image.shape[:2]
        
        # ROI cropping
        if self.top_roi_enabled and self.top_roi_width > 0 and self.top_roi_height > 0:
            roi_x = int(np.clip(self.top_roi_x, 0, width-1))
            roi_y = int(np.clip(self.top_roi_y, 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.top_roi_width, 0, width))
            roi_y2 = int(np.clip(roi_y + self.top_roi_height, 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
            roi_bbox = (roi_x, roi_y, roi_x2, roi_y2)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height
            roi_bbox = (0, 0, width, height)
        
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
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
                
                hand_lm = self.extract_hand_landmarks(hand_landmarks, width, height, roi_offset, roi_bbox)
                
                if handedness == "Left":
                    left_hand = hand_lm
                else:
                    right_hand = hand_lm

        # Stitch back into full image if ROI
        if self.top_roi_enabled and self.top_roi_width > 0 and self.top_roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        return full_annotated, left_hand, right_hand

    def extract_pose_landmarks(self, results, width, height, roi_offset, roi_bbox):
        """ランドマークを抽出（全画像座標）"""
        landmarks = []
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
                z = lm.z
                landmarks.extend([x, y, z])
        return landmarks

    def extract_face_landmarks(self, results, width, height, roi_offset, roi_bbox):
        """顔ランドマークを抽出"""
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
        """手ランドマークを抽出"""
        landmarks = []
        if hand_lm:
            for lm in hand_lm.landmark:
                x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
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