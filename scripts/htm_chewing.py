#!/usr/bin/env python3

import time
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32, String, Bool, Float32
from cv_bridge import CvBridge


# 1. CONFIGURATION & SENSITIVITY (Updated thresholds)

PEAK_THRESHOLD = 0.002     # Lower = More sensitive to small jaw movements
CHEWING_WINDOW = 30         # Frames to smooth the motion signal (removes jitter)

# Hand-to-Mouth thresholds (normalized distance 0.0-1.0)
HTM_START_THRESH = 0.30      # Hand must get this close to trigger "INTAKE"
HTM_END_THRESH = 0.40        # Hand must move this far to be considered "RETREATED"
VALIDATION_TIME = 3.0        # Seconds to wait for chewing after hand leaves face

# Landmark indices
FACIAL_LANDMARKS = list(range(0, 18)) + list(range(61, 88))  # Jaw & Lips
REFERENCE_LANDMARKS = [1]    # Nose Tip anchor
LEFT_WRIST = 15
RIGHT_WRIST = 16
MOUTH_CENTER = 13


class BiteCounter:
    """
    Extended logic:
    - total_bites: number of confirmed bites
    - current_chew_count: number of chewing peaks detected in current bite
    - last_bite_chews: chew count of the previous completed bite
    - Peak detection uses a simple 3-point pattern (rising then falling above threshold).
    """
    def __init__(self):
        self.state = "IDLE"
        self.total_bites = 0
        self.current_chew_count = 0
        self.last_bite_chews = 0
        self.last_intake_time = 0.0

        # For peak detection
        self.prev_val = 0.0
        self.prev_prev_val = 0.0

    def update(self, hand_dist: float, is_chewing: bool, current_motion_val: float):
        curr_time = time.time()

        # A. Peak detection (only when in active eating states)
        if self.state in ["CHEWING", "VALIDATION"] and is_chewing:
            # Peak pattern: previous > previous_previous AND previous > current AND previous above threshold
            if (self.prev_prev_val < self.prev_val) and (self.prev_val > current_motion_val) and (self.prev_val > PEAK_THRESHOLD):
                self.current_chew_count += 1

        # Update motion value history
        self.prev_prev_val = self.prev_val
        self.prev_val = current_motion_val

        # B. State machine transitions
        if self.state == "IDLE":
            if hand_dist < HTM_START_THRESH:
                self.state = "INTAKE"
                self.last_intake_time = curr_time
                self.current_chew_count = 0  # Prepare for potential new bite

        elif self.state == "INTAKE":
            if hand_dist > HTM_END_THRESH:
                self.state = "VALIDATION"
                self.last_intake_time = curr_time
            elif is_chewing:
                # Chewing started while hand still near mouth
                self._trigger_bite()

        elif self.state == "VALIDATION":
            if is_chewing:
                self._trigger_bite()
            elif (curr_time - self.last_intake_time) > VALIDATION_TIME:
                # False alarm (scratch etc.)
                self.state = "IDLE"
                self.current_chew_count = 0
            elif hand_dist < HTM_START_THRESH:
                self.state = "INTAKE"

        elif self.state == "CHEWING":
            if not is_chewing and hand_dist > HTM_END_THRESH:
                # Chewing ended and hand is away -> finalize bite
                self.last_bite_chews = self.current_chew_count
                self.current_chew_count = 0
                self.state = "IDLE"

        return self.total_bites, self.current_chew_count, self.last_bite_chews, self.state

    def _trigger_bite(self):
        if self.state != "CHEWING":
            self.total_bites += 1
            self.current_chew_count = 0  # Start counting fresh
            self.state = "CHEWING"


def get_distance(p1, p2) -> float:
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))


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
        super().__init__("eating_episode_detector")

        # Parameters
        self.declare_parameter("color_topic", "/front_camera/color/image_raw")
        self.declare_parameter("color_info_topic", "/front_camera/color/camera_info")
        self.declare_parameter("depth_topic", "/front_camera/depth/image_raw")
        self.declare_parameter("depth_info_topic", "/front_camera/depth/camera_info")
        self.declare_parameter("show_window", True)
        self.declare_parameter("annotated_image_topic", "~/annotated_image")

        self.color_topic = self.get_parameter("color_topic").value
        self.color_info_topic = self.get_parameter("color_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_info_topic = self.get_parameter("depth_info_topic").value
        self.show_window = self.get_parameter("show_window").value
        self.annotated_image_topic = self.get_parameter("annotated_image_topic").value

        self.bridge = CvBridge()

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

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

        self.get_logger().info("EatingEpisodeNode (chew peak version) initialized.")

    def color_info_cb(self, _: CameraInfo):
        pass

    def depth_image_cb(self, _: Image):
        pass

    def depth_info_cb(self, _: CameraInfo):
        pass

    def color_image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w = cv_image.shape[:2]
        cv_image.flags.writeable = False
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        cv_image.flags.writeable = True

        # Defaults
        is_chewing = False
        closest_hand_dist = 1.0
        avg_motion = 0.0

        # Face motion & chewing detection
        if results.face_landmarks:
            fm = results.face_landmarks
            coords = np.array([[p.x, p.y] for p in fm.landmark], dtype=np.float32)

            if self.prev_landmarks is not None:
                motion_mag = get_relative_motion(self.prev_landmarks, coords)
                self.motion_queue.append(motion_mag)
                self.motion_history.append(motion_mag)
                avg_motion = float(np.mean(self.motion_queue))
                is_chewing = avg_motion > PEAK_THRESHOLD

            self.prev_landmarks = coords.copy()
        else:
            is_chewing = False

        # Hand distance
        if results.pose_landmarks and results.face_landmarks:
            pose_lm = results.pose_landmarks.landmark
            face_lm = results.face_landmarks.landmark
            mouth_point = face_lm[MOUTH_CENTER]
            left_dist = get_distance(pose_lm[LEFT_WRIST], mouth_point)
            right_dist = get_distance(pose_lm[RIGHT_WRIST], mouth_point)
            closest_hand_dist = min(left_dist, right_dist)

        # Update bite counter logic (passes avg_motion for peak detection)
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

        # Publish results
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
            cv2.imshow("Eating Episode Detector (Chew Peaks)", overlay)
            cv2.waitKey(1)

    def draw_overlay(self, frame, w, h,
                     total_bites, current_chews, last_chews,
                     state, is_chewing, avg_motion, hand_dist):
        # Colors
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

        # Graph bottom-right
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
            if self.holistic:
                self.holistic.close()
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