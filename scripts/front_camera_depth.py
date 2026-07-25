#!/usr/bin/env python3

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

class FrontCameraNode(Node):
    def __init__(self):
        super().__init__('front_camera')

        # ==== CV Bridge ====
        self.bridge = CvBridge()

        # ==== MediaPipe ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh

        # ==== Parameters ====
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.6)
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0)
        self.declare_parameter('roi_width', 400)
        self.declare_parameter('roi_height', 300)

        # topics / frames
        self.declare_parameter('color_topic', '/front_camera/color/image_raw')
        self.declare_parameter('color_info_topic', '/front_camera/color/camera_info')
        self.declare_parameter('depth_topic', '/front_camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/front_camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'front_camera_depth_optical_frame')
        self.declare_parameter('publish_face_tf', False)  # 顔478点は重いので既定OFF
        self.declare_parameter('tf_rate_hz', 30.0)        # tfスロットリング

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
        self.publish_face_tf = bool(self.get_parameter('publish_face_tf').value)
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        # ==== MediaPipe Initializations ====
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

        # ==== ROI GUI ====
        self.dragging = False
        self.start_point = None
        self.end_point = None

        self.setup_opencv_window()

        # ==== Publishers ====
        self.annotated_pub = self.create_publisher(Image, '/front_camera/annotated_image', 10)
        self.pose_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/pose_landmarks', 10)
        self.face_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/face_landmarks', 10)
        self.left_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/left_hand_landmarks', 10)
        self.right_hand_landmarks_pub = self.create_publisher(Float32MultiArray, '/front_camera/right_hand_landmarks', 10)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_tf_time = self.get_clock().now()

        # ==== Subscribers with synchronization ====
        color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=10)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=10)
        depth_info_sub = message_filters.Subscriber(self, CameraInfo, self.depth_info_topic, qos_profile=10)
        # （カラーの CameraInfo も必要なら追加同期可。ここでは深度側を用いる前提：depth が color に整列済み）

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, depth_info_sub], queue_size=20, slop=0.05
        )
        ats.registerCallback(self.synced_callback)

        self.get_logger().info('Front Camera Node initialized (with depth→3D & tf broadcasting)')

    # ====================== GUI (ROI) ======================
    def setup_opencv_window(self):
        try:
            cv2.namedWindow('Front Camera - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Front Camera - ROI Selection', self.mouse_callback)
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
    def synced_callback(self, color_msg: CompressedImage, depth_msg: CompressedImage, depth_info: CameraInfo):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            if color is None:
                self.get_logger().error('Failed to decode color image')
                return
        except Exception as e:
            self.get_logger().error(f'Color decoding error ({type(e).__name__}): {e}')
            return

        try:
            # compressedDepth is a special ROS format, use cv_bridge with explicit handling
            # depth_header = depth_msg.data[:12]
            # depth_data = depth_msg.data[12:]

            # np_arr = np.frombuffer(depth_data, np.uint8)
            # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)

            # if cv_image is not None:
            #     valid_depth = cv_image[cv_image > 0]
            #     if len(valid_depth) > 0:
            #         self.get_logger().info(
            #             f"Depth Compressed - Min: {valid_depth.min():.3f}m, "
            #             f"Max: {valid_depth.max():.3f}m, Mean: {valid_depth.mean():.3f}m, "
            #             f"Format: {depth_msg.format}, Size: {len(depth_msg.data)} bytes"
            #         )
            
            #     depth_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            #     depth_color = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
            #     cv2.imshow('/depth/image_raw/compressed', depth_color)
            #     cv2.waitKey(1)

            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth is None:
                self.get_logger().error('Failed to decode depth image')
                return
            
            # Normalize to meters based on dtype
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0  # mm → m
            elif depth.dtype == np.float32:
                depth_m = depth.astype(np.float32)  # already in meters
            else:
                # Try to interpret as uint16 if unknown
                depth_m = depth.astype(np.float32) / 1000.0
                self.get_logger().warn(f'Unknown depth dtype: {depth.dtype}, assuming mm units')
        except Exception as e:
            self.get_logger().error(f'Depth decoding error ({type(e).__name__}): {e}')
            return

        annotated_image, pose_lm, face_lm, lhand_lm, rhand_lm, roi_ctx = self.process_image(color)

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.annotated_pub.publish(ann)

        # Publish 2D landmarks (pixel + mediapipe z)
        self._publish_array(self.pose_landmarks_pub, pose_lm)
        self._publish_array(self.face_landmarks_pub, face_lm)
        self._publish_array(self.left_hand_landmarks_pub, lhand_lm)
        self._publish_array(self.right_hand_landmarks_pub, rhand_lm)

        # 3D projection & TF
        # Use depth intrinsics (assuming depth is registered to color)
        fx = depth_info.k[0]; fy = depth_info.k[4]
        cx = depth_info.k[2]; cy = depth_info.k[5]

        # Validate intrinsics
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error(f'Invalid intrinsics: fx={fx}, fy={fy}')
            return

        # rate limit TF
        now = self.get_clock().now()
        if (now - self.last_tf_time).nanoseconds < (1e9 / self.tf_rate_hz):
            return
        self.last_tf_time = now

        # ROI offset to full image coords
        roi_x, roi_y, roi_w, roi_h, roi_enabled = roi_ctx

        # Helper to loop & broadcast
        def broadcast_set(flat_xyz, prefix):
            # flat list [x_pix, y_pix, z_mp, ...]
            n = len(flat_xyz) // 3
            for i in range(n):
                u = float(flat_xyz[3*i + 0])
                v = float(flat_xyz[3*i + 1])
                # clamp to image bounds
                u_i = int(np.clip(u, 0, depth_m.shape[1]-1))
                v_i = int(np.clip(v, 0, depth_m.shape[0]-1))
                z = self._robust_depth(depth_m, v_i, u_i)  # meters

                if not np.isfinite(z) or z <= 0.0:
                    continue

                X = (u - cx) / fx * z
                Y = (v - cy) / fy * z
                # camera optical frame: X right, Y down, Z forward (ROS REP 103 optical)
                # Transform with identity rotation; only translation
                t = TransformStamped()
                t.header.stamp = color_msg.header.stamp
                t.header.frame_id = self.camera_frame
                t.child_frame_id = f'{prefix}_{i}'
                t.transform.translation.x = float(X)
                t.transform.translation.y = float(Y)
                t.transform.translation.z = float(z)
                # no rotation (identity)
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

        # Note: pose=33, hands=21 each, face=478(任意)
        if pose_lm:
            broadcast_set(pose_lm, 'front_camera_pose')
        if lhand_lm:
            broadcast_set(lhand_lm, 'front_camera_left_hand')
        if rhand_lm:
            broadcast_set(rhand_lm, 'front_camera_right_hand')
        if self.publish_face_tf and face_lm:
            broadcast_set(face_lm, 'front_camera_face')

        # Show ROI helper window
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
        cv2.imshow('Front Camera - ROI Selection', disp)
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
        """3x3 median (ignore zeros) → meters"""
        h, w = depth_m.shape
        v0 = max(0, v-1); v1 = min(h, v+2)
        u0 = max(0, u-1); u1 = min(w, u+2)
        patch = depth_m[v0:v1, u0:u1].reshape(-1)
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def process_image(self, cv_image):
        """Run MediaPipe on ROI (if enabled), draw, and return full annotated image + flat landmark lists."""
        height, width = cv_image.shape[:2]

        # ROI crop
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
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        # Extract landmarks in full-image pixel coords (x_pix, y_pix, z_mp)
        pose_landmarks = self.extract_pose_landmarks(holistic_results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        face_landmarks = self.extract_face_landmarks(face_results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        left_hand_landmarks = self.extract_hand_landmarks(holistic_results.left_hand_landmarks, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))
        right_hand_landmarks = self.extract_hand_landmarks(holistic_results.right_hand_landmarks, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))

        roi_ctx = (self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.roi_enabled)
        return full_annotated, pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks, roi_ctx

    def extract_pose_landmarks(self, results, width, height, roi_offset, roi_bbox):
        landmarks = []
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = lm.x * (roi_bbox[2] - roi_bbox[0]) + roi_offset[0]
                y = lm.y * (roi_bbox[3] - roi_bbox[1]) + roi_offset[1]
                z = lm.z  # MediaPipeの相対Z（参考までに保持）
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
    node = FrontCameraNode()
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