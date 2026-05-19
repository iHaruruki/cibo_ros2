# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import message_filters
from rclpy.parameter import Parameter

class TopCameraNode(Node):
    def __init__(self):
        super().__init__('top_camera')

        # ==== CV Bridge ====
        self.bridge = CvBridge()

        # ==== MediaPipe ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # ==== Parameters ====
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.6)
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0)
        self.declare_parameter('roi_width', 400)
        self.declare_parameter('roi_height', 300)

        # topics / frames
        self.declare_parameter('color_topic', '/top_camera/color/image_raw')
        self.declare_parameter('color_info_topic', '/top_camera/color/camera_info')
        self.declare_parameter('depth_topic', '/top_camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/top_camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'top_camera_depth_optical_frame')
        self.declare_parameter('tf_rate_hz', 30.0)

        # Read params
        min_det = float(self.get_parameter('min_detection_confidence').value)
        min_trk = float(self.get_parameter('min_tracking_confidence').value)
        self.roi_enabled = bool(self.get_parameter('roi_enabled').value)
        self.roi_x = int(self.get_parameter('roi_x').value)
        self.roi_y = int(self.get_parameter('roi_y').value)
        self.roi_width = int(self.get_parameter('roi_width').value)
        self.roi_height = int(self.get_parameter('roi_height').value)

        self.color_topic = self.get_parameter('color_topic').value
        self.color_info_topic = self.get_parameter('color_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        # ==== MediaPipe Initializations ====
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

        # ==== ROI GUI ====
        self.dragging = False
        self.start_point = None
        self.end_point = None

        self.setup_opencv_window()

        # ==== Publishers ====
        self.annotated_pub = self.create_publisher(Image, '/top_camera/annotated_image', 10)
        self.left_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/top_camera/left_hand_landmarks', 10)
        self.right_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/top_camera/right_hand_landmarks', 10)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_tf_time = self.get_clock().now()

        # ==== Subscribers with synchronization ====
        color_sub = message_filters.Subscriber(self, CompressedImage, self.color_topic, qos_profile=10)
        depth_sub = message_filters.Subscriber(self, CompressedImage, self.depth_topic, qos_profile=10)
        depth_info_sub = message_filters.Subscriber(self, CameraInfo, self.depth_info_topic, qos_profile=10)

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, depth_info_sub], queue_size=20, slop=0.05
        )
        ats.registerCallback(self.synced_callback)

        self.get_logger().info('Top Camera Node initialized (hands only, with depth→3D & tf broadcasting)')

    # ====================== GUI (ROI) ======================
    def setup_opencv_window(self):
        try:
            cv2.namedWindow('Top Camera - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Top Camera - ROI Selection', self.mouse_callback)
            self.get_logger().info('OpenCV window setup for ROI selection')
        except Exception as e:
            self.get_logger().error(f'Failed to setup OpenCV window: {str(e)}')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.start_point:
                self.dragging = False
                self.end_point = (x, y)
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                self.roi_x = x1
                self.roi_y = y1
                self.roi_width = x2 - x1
                self.roi_height = y2 - y1
                self.roi_enabled = True
                self.set_parameters([
                    Parameter('roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('roi_x', Parameter.Type.INTEGER, self.roi_x),
                    Parameter('roi_y', Parameter.Type.INTEGER, self.roi_y),
                    Parameter('roi_width', Parameter.Type.INTEGER, self.roi_width),
                    Parameter('roi_height', Parameter.Type.INTEGER, self.roi_height),
                ])
                self.get_logger().info(f'ROI set: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}')

    # ====================== Core ======================
    def synced_callback(self, color_msg: Image, depth_msg: Image, depth_info: CameraInfo):
        try:
            np_arr = np.frombuffer(color_msg, np.uint8)
            color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'color cv bridge error: {e}')
            return

        # depth image decoding
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            # normalize units to meters
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_m = depth.astype(np.float32) / 1000.0  # mm → m
            elif depth_msg.encoding in ('32FC1'):
                depth_m = depth.astype(np.float32)
            else:
                depth_m = depth.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'depth cv bridge error: {e}')
            return

        annotated_image, lhand_lm, rhand_lm, roi_ctx = self.process_image(color)

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.annotated_pub.publish(ann)

        # Publish 2D landmarks (pixel + mediapipe z)
        self._publish_array(self.left_hand_landmarks_pub, lhand_lm)
        self._publish_array(self.right_hand_landmarks_pub, rhand_lm)

        # 3D projection & TF
        fx = depth_info.k[0]; fy = depth_info.k[4]
        cx = depth_info.k[2]; cy = depth_info.k[5]

        now = self.get_clock().now()
        if (now - self.last_tf_time).nanoseconds < (1e9 / self.tf_rate_hz):
            return
        self.last_tf_time = now

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
                t.header.frame_id = self.camera_frame
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
            broadcast_set(lhand_lm, 'top_camera_right_hand')
        if rhand_lm:
            broadcast_set(rhand_lm, 'top_camera_left_hand')

        disp = annotated_image.copy()
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            cv2.rectangle(disp, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_width, self.roi_y + self.roi_height), (0, 255, 0), 2)
            cv2.putText(disp, 'ROI', (self.roi_x, self.roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.dragging and self.start_point and self.end_point:
            cv2.rectangle(disp, self.start_point, self.end_point, (255, 0, 0), 2)
            cv2.putText(disp, 'Selecting ROI...',
                        (self.start_point[0], self.start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(disp, 'Drag to select ROI', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Top Camera - ROI Selection', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
        elif key == ord('r'):
            self.roi_enabled = False
            self.set_parameters([Parameter('roi_enabled', Parameter.Type.BOOL, False)])
            self.get_logger().info('ROI reset')

    # ---------- helpers ----------
    def _publish_array(self, pub, flat):
        msg = Float32MultiArray()
        msg.data = flat
        pub.publish(msg)

    def _robust_depth(self, depth_m, v, u):
        h, w = depth_m.shape
        v0 = max(0, v-1); v1 = min(h, v+2)
        u0 = max(0, u-1); u1 = min(w, u+2)
        patch = depth_m[v0:v1, u0:u1].reshape(-1)
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def process_image(self, cv_image):
        height, width = cv_image.shape[:2]
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            roi_x = int(np.clip(self.roi_x, 0, width-1))
            roi_y = int(np.clip(self.roi_y, 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.roi_width, 0, width))
            roi_y2 = int(np.clip(roi_y + self.roi_height, 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height

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

        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        roi_ctx = (self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.roi_enabled)
        return full_annotated, left_hand_landmarks, right_hand_landmarks, roi_ctx

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
    node = TopCameraNode()
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