#!/usr/bin/env python3

import time
from collections import deque
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32, String, Bool, Float32
from cv_bridge import CvBridge


# =========================
# 1) CONFIGURATION
# =========================
PEAK_THRESHOLD = 0.002
CHEWING_WINDOW = 30

HTM_START_THRESH = 0.30
HTM_END_THRESH = 0.40
VALIDATION_TIME = 3.0

# Face landmarks (same intent as original)
FACIAL_LANDMARKS = list(range(0, 18)) + list(range(61, 88))
REFERENCE_LANDMARKS = [1]  # Nose tip-ish anchor in face mesh topology
MOUTH_CENTER = 13


class BiteCounter:
    def __init__(self):
        self.state = "IDLE"
        self.total_bites = 0
        self.current_chew_count = 0
        self.last_bite_chews = 0
        self.last_intake_time = 0.0

        self.prev_val = 0.0
        self.prev_prev_val = 0.0

    def update(self, hand_dist: float, is_chewing: bool, current_motion_val: float):
        curr_time = time.time()

        if self.state in ["CHEWING", "VALIDATION"] and is_chewing:
            if (self.prev_prev_val < self.prev_val) and (self.prev_val > current_motion_val) and (self.prev_val > PEAK_THRESHOLD):
                self.current_chew_count += 1

        self.prev_prev_val = self.prev_val
        self.prev_val = current_motion_val

        if self.state == "IDLE":
            if hand_dist < HTM_START_THRESH:
                self.state = "INTAKE"
                self.last_intake_time = curr_time
                self.current_chew_count = 0

        elif self.state == "INTAKE":
            if hand_dist > HTM_END_THRESH:
                self.state = "VALIDATION"
                self.last_intake_time = curr_time
            elif is_chewing:
                self._trigger_bite()

        elif self.state == "VALIDATION":
            if is_chewing:
                self._trigger_bite()
            elif (curr_time - self.last_intake_time) > VALIDATION_TIME:
                self.state = "IDLE"
                self.current_chew_count = 0
            elif hand_dist < HTM_START_THRESH:
                self.state = "INTAKE"

        elif self.state == "CHEWING":
            if not is_chewing and hand_dist > HTM_END_THRESH:
                self.last_bite_chews = self.current_chew_count
                self.current_chew_count = 0
                self.state = "IDLE"

        return self.total_bites, self.current_chew_count, self.last_bite_chews, self.state

    def _trigger_bite(self):
        if self.state != "CHEWING":
            self.total_bites += 1
            self.current_chew_count = 0
            self.state = "CHEWING"


def get_relative_motion(prev_coords: np.ndarray, curr_coords: np.ndarray) -> float:
    ref_prev = np.mean(prev_coords[REFERENCE_LANDMARKS], axis=0)
    ref_curr = np.mean(curr_coords[REFERENCE_LANDMARKS], axis=0)

    rel_prev = prev_coords[FACIAL_LANDMARKS] - ref_prev
    rel_curr = curr_coords[FACIAL_LANDMARKS] - ref_curr

    delta = rel_curr - rel_prev
    mag = np.linalg.norm(delta, axis=1)
    return float(np.mean(mag))


def draw_text_with_outline(img, text, org, font, font_scale, color, thickness, outline_color=(0, 0, 0)):
    cv2.putText(img, text, org, font, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


class EatingEpisodeNode(Node):
    def __init__(self):
        super().__init__("eating_episode_detector_tasks")

        # ROS params
        self.declare_parameter("color_topic", "/front_camera/color/image_raw")
        self.declare_parameter("color_info_topic", "/front_camera/color/camera_info")
        self.declare_parameter("depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("depth_info_topic", "/front_camera/depth/camera_info")
        self.declare_parameter("show_window", True)
        self.declare_parameter("annotated_image_topic", "/annotated_image")

        # New model-asset params (MediaPipe Tasks)
        self.declare_parameter("face_model_path", "models/face_landmarker.task")
        self.declare_parameter("hand_model_path", "models/hand_landmarker.task")

        self.color_topic = self.get_parameter("color_topic").value
        self.color_info_topic = self.get_parameter("color_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_info_topic = self.get_parameter("depth_info_topic").value
        self.show_window = self.get_parameter("show_window").value
        self.annotated_image_topic = self.get_parameter("annotated_image_topic").value
        self.face_model_path = self.get_parameter("face_model_path").value
        self.hand_model_path = self.get_parameter("hand_model_path").value

        self.bridge = CvBridge()

        # ----- MediaPipe Tasks setup -----
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.mp_vision = mp_vision
        self._ts_ms = 0

        face_base = mp_python.BaseOptions(model_asset_path=self.face_model_path)
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

        hand_base = mp_python.BaseOptions(model_asset_path=self.hand_model_path)
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # State
        self.bite_counter = BiteCounter()
        self.prev_landmarks: Optional[np.ndarray] = None
        self.motion_queue: deque = deque(maxlen=CHEWING_WINDOW)
        self.motion_history: deque = deque(maxlen=200)

        # Publishers
        self.pub_bites = self.create_publisher(Int32, "~/bites_count", 10)
        self.pub_current_chews = self.create_publisher(Int32, "~/current_chew_count", 10)
        self.pub_last_bite_chews = self.create_publisher(Int32, "~/last_bite_chews", 10)
        self.pub_state = self.create_publisher(String, "~/state", 10)
        self.pub_is_chewing = self.create_publisher(Bool, "~/is_chewing", 10)
        self.pub_hand_dist = self.create_publisher(Float32, "~/hand_distance", 10)
        self.pub_motion_val = self.create_publisher(Float32, "~/motion_value", 10)
        self.pub_annotated = self.create_publisher(Image, self.annotated_image_topic, 10)

        # Subscriptions
        self.sub_color = self.create_subscription(Image, self.color_topic, self.color_image_cb, 10)
        self.sub_color_info = self.create_subscription(CameraInfo, self.color_info_topic, self.color_info_cb, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_image_cb, 10)
        self.sub_depth_info = self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_cb, 10)

        self.get_logger().info("EatingEpisodeNode (MediaPipe Tasks API) initialized.")

    def color_info_cb(self, _: CameraInfo):
        pass

    def depth_image_cb(self, _: Image):
        pass

    def depth_info_cb(self, _: CameraInfo):
        pass

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.time() * 1000)
        if now_ms <= self._ts_ms:
            now_ms = self._ts_ms + 1
        self._ts_ms = now_ms
        return self._ts_ms

    @staticmethod
    def _norm_dist(p1, p2) -> float:
        return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))

    def _extract_face_coords(self, face_result) -> Optional[np.ndarray]:
        if not face_result.face_landmarks:
            return None
        # first face only
        lm = face_result.face_landmarks[0]
        coords = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        return coords

    def _extract_mouth_point(self, face_result):
        if not face_result.face_landmarks:
            return None
        lm = face_result.face_landmarks[0]
        if len(lm) <= MOUTH_CENTER:
            return None
        return lm[MOUTH_CENTER]

    def _closest_hand_to_point(self, hand_result, point) -> float:
        if point is None or not hand_result.hand_landmarks:
            return 1.0

        # Wrist index in hand landmark topology = 0
        closest = 1.0
        for hand_lms in hand_result.hand_landmarks:
            wrist = hand_lms[0]
            d = self._norm_dist(wrist, point)
            if d < closest:
                closest = d
        return closest

    def color_image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w = cv_image.shape[:2]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = self._next_timestamp_ms()

        # Tasks inference
        face_result = self.face_landmarker.detect_for_video(mp_image, ts_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts_ms)

        is_chewing = False
        closest_hand_dist = 1.0
        avg_motion = 0.0

        # Face motion
        face_coords = self._extract_face_coords(face_result)
        if face_coords is not None:
            if self.prev_landmarks is not None:
                motion_mag = get_relative_motion(self.prev_landmarks, face_coords)
                self.motion_queue.append(motion_mag)
                self.motion_history.append(motion_mag)
                avg_motion = float(np.mean(self.motion_queue))
                is_chewing = avg_motion > PEAK_THRESHOLD
            self.prev_landmarks = face_coords.copy()
        else:
            is_chewing = False

        # Hand-to-mouth
        mouth_point = self._extract_mouth_point(face_result)
        closest_hand_dist = self._closest_hand_to_point(hand_result, mouth_point)

        # Update state machine
        total_bites, current_chews, last_chews, state = self.bite_counter.update(
            closest_hand_dist, is_chewing, avg_motion
        )

        # Visualization
        overlay = cv_image.copy()
        self.draw_overlay(
            overlay, w, h,
            total_bites, current_chews, last_chews,
            state, is_chewing, avg_motion, closest_hand_dist
        )

        # Publish
        self.publish_metrics(
            total_bites=total_bites,
            current_chews=current_chews,
            last_chews=last_chews,
            state=state,
            is_chewing=is_chewing,
            hand_dist=closest_hand_dist,
            motion_val=avg_motion,
            header=msg.header,
            annotated=overlay
        )

        if self.show_window:
            cv2.imshow("Eating Episode Detector (Tasks API)", overlay)
            cv2.waitKey(1)

    def draw_overlay(self, frame, w, h,
                     total_bites, current_chews, last_chews,
                     state, is_chewing, avg_motion, hand_dist):
        COLOR_MAIN = (0, 255, 0)
        COLOR_SECONDARY = (255, 255, 0)
        COLOR_ALERT = (0, 165, 255)
        COLOR_SUCCESS = (0, 0, 255)

        display_chews = current_chews if state != "IDLE" else last_chews
        chew_color = COLOR_SUCCESS if state != "IDLE" else COLOR_SECONDARY
        state_display_color = COLOR_SECONDARY
        if state == "INTAKE":
            state_display_color = COLOR_ALERT
        elif state == "CHEWING":
            state_display_color = COLOR_SUCCESS

        draw_text_with_outline(frame, f"BITES: {total_bites}", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_MAIN, 2)
        draw_text_with_outline(frame, f"Chews: {display_chews}", (180, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, chew_color, 2)
        draw_text_with_outline(frame, f"State: {state}", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_display_color, 1)
        chew_status = "YES" if is_chewing else "NO"
        draw_text_with_outline(frame, f"Motion: {chew_status} ({avg_motion:.4f})", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAIN, 1)
        draw_text_with_outline(frame, f"Hand Dist: {hand_dist:.2f}", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAIN, 1)

        graph_h, graph_w = 100, 320
        graph_x_start = w - graph_w - 10
        graph_y_start = h - graph_h - 10
        if graph_x_start >= 0 and graph_y_start >= 0:
            graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
            if len(self.motion_history) > 1:
                mh = np.array(self.motion_history)
                mh_min, mh_max = float(mh.min()), float(mh.max())
                mh_norm = (mh - mh_min) / (mh_max - mh_min + 1e-6)
                points = [
                    (int(i * graph_w / len(mh_norm)), int(graph_h * (1 - v)))
                    for i, v in enumerate(mh_norm)
                ]
                for i in range(1, len(points)):
                    cv2.line(graph, points[i - 1], points[i], (0, 255, 255), 1)

                norm_thresh = (PEAK_THRESHOLD - mh_min) / (mh_max - mh_min + 1e-6)
                thresh_y = int(graph_h * (1 - norm_thresh))
                if 0 <= thresh_y < graph_h:
                    cv2.line(graph, (0, thresh_y), (graph_w, thresh_y), (0, 0, 255), 1)

            frame[graph_y_start:graph_y_start + graph_h, graph_x_start:graph_x_start + graph_w] = graph

    def publish_metrics(self, total_bites, current_chews, last_chews,
                        state, is_chewing, hand_dist, motion_val,
                        header, annotated):
        self.pub_bites.publish(Int32(data=int(total_bites)))
        self.pub_current_chews.publish(Int32(data=int(current_chews)))
        self.pub_last_bite_chews.publish(Int32(data=int(last_chews)))
        self.pub_state.publish(String(data=str(state)))
        self.pub_is_chewing.publish(Bool(data=bool(is_chewing)))
        self.pub_hand_dist.publish(Float32(data=float(hand_dist)))
        self.pub_motion_val.publish(Float32(data=float(motion_val)))

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = header
            self.pub_annotated.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Annotated image publish failed: {e}")

    def destroy_node(self):
        try:
            if self.face_landmarker:
                self.face_landmarker.close()
        except Exception:
            pass
        try:
            if self.hand_landmarker:
                self.hand_landmarker.close()
        except Exception:
            pass
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EatingEpisodeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()