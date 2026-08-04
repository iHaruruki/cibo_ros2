#!/usr/bin/env python3

import os
import time
import math
import threading
import queue
import logging
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque

import cv2
import numpy as np
import requests

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Models
from ultralytics import YOLOWorld, SAM
import supervision as sv

# ------------------------- CONFIG -------------------------
API_KEY = os.getenv("USDA_API_KEY", "")
LOG_LEVEL = logging.INFO

YOLO_MODEL_PATH = "yolov8s-world.pt"
SAM_MODEL_PATH = "sam_b.pt"

# System Params
INTERACTION_IOU_THRESHOLD = 0.15
INTERACTION_COOLDOWN = 2.0
SEGMENTATION_WORKER_COUNT = 1
TRACK_EVICT_SECONDS = 60.0
DEPTH_INVALID = 0

# Bite Logic Params
VOL_BUFFER_LEN = 15          # moving average length
SETTLING_TIME = 1.5          # seconds
MIN_BITE_GRAMS = 5.0         # minimum delta to count as bite

# Detection config
DETECT_CONF = 0.12
DETECT_IOU = 0.45

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------- DATA & CLASSES -------------------------

# Utensils
UTENSIL_CLASSES = {"spoon", "fork", "knife", "chopsticks", "spork"}

# Full Class List
my_food_classes = [
    "human hand",
    "spoon", "fork", "knife", "chopsticks", "spork",

    "bowl of white rice",
    "bowl of brown rice",
    "bread slice",
    "bagel",
    "muffin",
    "toast",
    "pita bread",
    "pasta (serving)",
    "noodle dish",

    "steak (cut)",
    "chicken breast (cooked)",
    "burger (assembled)",
    "hot dog",
    "sausage",
    "salmon fillet",
    "fish fillet (cooked)",
    "eggs (cooked)",

    "sushi roll (slice)",
    "bowl of ramen",
    "bowl of pho",
    "dumpling (cooked)",
    "rice ball (onigiri)",
    "tofu (cube)",
    "curry bowl",
    "stir fry dish",
    "spring roll",

    "apple",
    "banana",
    "orange",
    "grapes (bunch)",
    "mango",
    "strawberry",
    "avocado",
    "lemon",

    "broccoli (floret)",
    "carrot (piece)",
    "potato (baked/boiled)",
    "sweet potato",
    "tomato",
    "corn cob",
    "lettuce",
    "cabbage (wedge)",

    "pizza slice",
    "french fries (serving)",
    "taco",
    "sandwich",
    "bowl of soup",
    "bowl of chili",
    "salad bowl",

    "donut",
    "cookie",
    "slice of cake",
    "ice cream scoop",
    "pancake",
    "waffle",
    "yogurt bowl",
]

# Densities (g per mL)
DENSITY_DB: Dict[str, float] = {
    "default": 1.0,  # フォールバック
    "bowl of white rice": 0.80,
    "bowl of brown rice": 0.85,
    "bread slice": 0.35,
    "bagel": 0.45,
    "muffin": 0.50,
    "toast": 0.25,
    "pita bread": 0.40,
    "pasta (serving)": 0.90,
    "noodle dish": 0.80,

    "steak (cut)": 1.05,
    "chicken breast (cooked)": 1.02,
    "burger (assembled)": 0.85,
    "hot dog": 0.95,
    "sausage": 0.90,
    "salmon fillet": 1.04,
    "fish fillet (cooked)": 1.00,
    "eggs (cooked)": 1.03,

    "sushi roll (slice)": 1.05,
    "bowl of ramen": 0.80,
    "bowl of pho": 0.75,
    "dumpling (cooked)": 0.80,
    "rice ball (onigiri)": 1.10,
    "tofu (cube)": 1.00,
    "curry bowl": 0.95,
    "stir fry dish": 0.90,
    "spring roll": 0.60,

    "apple": 0.85,
    "banana": 0.90,
    "orange": 0.95,
    "grapes (bunch)": 1.05,
    "mango": 1.00,
    "strawberry": 0.90,
    "avocado": 0.95,
    "lemon": 1.00,

    "broccoli (floret)": 0.75,
    "carrot (piece)": 0.95,
    "potato (baked/boiled)": 0.80,
    "sweet potato": 0.90,
    "tomato": 0.93,
    "corn cob": 0.90,
    "lettuce": 0.15,
    "cabbage (wedge)": 0.60,

    "pizza slice": 0.70,
    "french fries (serving)": 0.40,
    "taco": 0.65,
    "sandwich": 0.45,
    "bowl of soup": 1.00,
    "bowl of chili": 1.05,
    "salad bowl": 0.20,

    "donut": 0.30,
    "cookie": 0.60,
    "slice of cake": 0.45,
    "ice cream scoop": 0.55,
    "pancake": 0.40,
    "waffle": 0.35,
    "yogurt bowl": 1.03
}

def get_density(label: str) -> float:
    return DENSITY_DB.get(label, DENSITY_DB["default"])

# ------------------------- STATE MANAGEMENT -------------------------
class FoodState:
    """Per-tracked-object state with smoothing and bite logic."""
    def __init__(self, label: str):
        self.label = label
        self.last_seen = time.time()
        self.vol_buffer: Deque[float] = deque(maxlen=VOL_BUFFER_LEN)
        self.current_avg_vol_ml = 0.0
        self.current_weight_g = 0.0
        # Bite state machine
        self.is_interacting = False
        self.interaction_start_time = 0.0
        self.pre_interaction_vol_ml = 0.0
        self.settling_start_time = 0.0
        self.waiting_for_settle = False

    def update_volume(self, vol_ml: float, density: float):
        if vol_ml > 0:
            self.vol_buffer.append(vol_ml)
            self.current_avg_vol_ml = sum(self.vol_buffer) / len(self.vol_buffer)
            self.current_weight_g = self.current_avg_vol_ml * density

TRACKED_OBJECT_CACHE: Dict[int, FoodState] = {}
current_interaction_display = {"text": "", "timer": 0.0}

# ------------------------- UTILITIES -------------------------
def get_manual_intrinsics(width=640, height=480, h_fov_deg=60.0):
    f_x = width / (2 * np.tan(np.deg2rad(h_fov_deg / 2)))
    f_y = f_x
    c_x = width / 2.0
    c_y = height / 2.0
    return {"fx": f_x, "fy": f_y, "cx": c_x, "cy": c_y}

def deproject_pixel_to_point(pixel: Tuple[int, int], depth_mm: float, intr: Dict) -> Tuple[float, float, float]:
    if depth_mm <= 0:
        return (float("nan"), float("nan"), float("nan"))
    z = depth_mm / 1000.0
    x = (pixel[0] - intr["cx"]) * z / intr["fx"]
    y = (pixel[1] - intr["cy"]) * z / intr["fy"]
    return (x, y, z)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def estimate_volume_from_mask(mask: np.ndarray, depth_map: np.ndarray, intr: Dict) -> float:
    """
    簡易ボクセル近似:
    - マスク内のDepth分布から床(テーブル)を95パーセンタイルで推定
    - ピクセル面積は平均深度とfx,fyから近似
    - 体積 = sum((table_depth - pixel_depth)+ * pixel_area)
    """
    try:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0.0
        depths = depth_map[ys, xs].astype(np.float32)
        valid = depths > DEPTH_INVALID
        if not np.any(valid):
            return 0.0
        depths = depths[valid]
        table_depth_mm = float(np.percentile(depths, 95))
        avg_d = float(np.mean(depths))
        # ピクセル面積近似（m^2）
        pixel_area_m2 = max(1e-10, (avg_d/1000.0)/intr["fx"] * (avg_d/1000.0)/intr["fy"])
        heights_m = np.maximum(0.0, (table_depth_mm - depths) / 1000.0)
        vol_m3 = float(np.sum(heights_m) * pixel_area_m2)
        return vol_m3 * 1e6  # mL
    except Exception:
        logging.exception("Volume estimation error")
        return 0.0

# ------------------------- ROS 2 NODE -------------------------
class FoodVolumeNode(Node):
    def __init__(self):
        super().__init__("food_volume_node")
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter("color_topic", "/camera_02/color/image_raw")
        self.declare_parameter("color_info_topic", "/camera_02/color/camera_info")
        self.declare_parameter("depth_topic", "/camera_02/depth/image_raw")
        self.declare_parameter("depth_info_topic", "/camera_02/depth/camera_info")

        self.color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        self.color_info_topic = self.get_parameter("color_info_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value

        self.get_logger().info(
            f"Subscribing to:\n  color: {self.color_topic}\n  color info: {self.color_info_topic}\n  depth: {self.depth_topic}\n  depth info: {self.depth_info_topic}"
        )

        # Subscribers
        self.sub_color = self.create_subscription(Image, self.color_topic, self.on_color, 10)
        self.sub_color_info = self.create_subscription(CameraInfo, self.color_info_topic, self.on_color_info, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.sub_depth_info = self.create_subscription(CameraInfo, self.depth_info_topic, self.on_depth_info, 10)

        # Buffers
        self.last_color: Optional[np.ndarray] = None
        self.last_color_stamp = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_depth_stamp = None

        # Intrinsics
        self.intrinsics: Dict = get_manual_intrinsics(640, 480, h_fov_deg=60.0)
        self.have_intrinsics = False

        # Models
        self.get_logger().info("Loading models...")
        self.detection_model = YOLOWorld(YOLO_MODEL_PATH)
        self.segmentation_model = SAM(SAM_MODEL_PATH)
        # ByteTrack version differences: use default ctor
        self.tracker = sv.ByteTrack()
        self.detection_model.set_classes(my_food_classes)

        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

        # Segmentation worker infra
        self.seg_job_queue: "queue.Queue[Tuple[np.ndarray, np.ndarray, List[int]]]" = queue.Queue(maxsize=4)
        self.seg_result_queue: "queue.Queue[Tuple[List[np.ndarray], List[int]]]" = queue.Queue(maxsize=8)
        self.seg_worker_stop = threading.Event()
        for _ in range(max(1, SEGMENTATION_WORKER_COUNT)):
            t = threading.Thread(target=self.segmentation_worker, daemon=True)
            t.start()

        # Processing timer
        self.timer = self.create_timer(0.03, self.process_once)
        self.last_cleanup = time.time()

    # CameraInfo
    def on_color_info(self, msg: CameraInfo):
        if not self.have_intrinsics:
            self.update_intrinsics_from_msg(msg)

    def on_depth_info(self, msg: CameraInfo):
        if not self.have_intrinsics:
            self.update_intrinsics_from_msg(msg)

    def update_intrinsics_from_msg(self, msg: CameraInfo):
        try:
            fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            if fx > 0 and fy > 0:
                self.intrinsics = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
                self.have_intrinsics = True
                self.get_logger().info(f"Updated intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    # Image callbacks
    def on_color(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_color = cv_img
            self.last_color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")

    def on_depth(self, msg: Image):
        try:
            if msg.encoding in ["16UC1"]:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.uint16)
            elif msg.encoding in ["32FC1"]:
                depth_f = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
                depth = (np.clip(depth_f, 0, np.finfo(np.float32).max) * 1000.0).astype(np.uint16)
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if depth.dtype != np.uint16:
                    depth = (depth.astype(np.float32) * 1000.0).astype(np.uint16)
            self.last_depth = depth
            self.last_depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    # Segmentation worker thread
    def segmentation_worker(self):
        logging.info("Segmentation worker started")
        while not self.seg_worker_stop.is_set():
            try:
                job = self.seg_job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            color_image, boxes, tracker_ids = job
            try:
                if len(boxes) == 0:
                    self.seg_result_queue.put(([], tracker_ids))
                    self.seg_job_queue.task_done()
                    continue
                res = self.segmentation_model.predict(source=color_image, bboxes=boxes, verbose=False)
                masks = []
                try:
                    out0 = res[0]
                    if hasattr(out0, "masks") and out0.masks is not None:
                        # Ultralytics SAM: .masks.data is torch.Tensor [N,H,W]
                        data = getattr(out0.masks, "data", None)
                        if data is not None:
                            masks_np = data.detach().cpu().numpy()
                            # binarize
                            masks = [(m > 0.5).astype(np.uint8) for m in masks_np]
                        else:
                            # Some versions expose .masks as numpy already
                            arr = np.array(out0.masks, dtype=np.uint8)
                            if arr.ndim == 3:
                                masks = [(m > 0).astype(np.uint8) for m in arr]
                    else:
                        # Fallback via supervision
                        try:
                            sam_dets = sv.Detections.from_ultralytics(out0)
                            masks = [m.astype(np.uint8) for m in getattr(sam_dets, "mask", []) if m is not None]
                        except Exception:
                            masks = []
                except Exception:
                    masks = []

                # Ensure mask list length matches tracker_ids length when possible
                self.seg_result_queue.put((masks, tracker_ids))
            except Exception:
                logging.exception("Error during segmentation")
                self.seg_result_queue.put(([], tracker_ids))
            finally:
                self.seg_job_queue.task_done()

    # Processing loop
    def process_once(self):
        if self.last_color is None or self.last_depth is None:
            return

        color_image = self.last_color
        depth_map = self.last_depth

        # Detection
        try:
            results = self.detection_model.predict(color_image, conf=DETECT_CONF, iou=DETECT_IOU, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
        except Exception:
            logging.exception("Detection failed")
            detections = sv.Detections.empty()

        class_names = detections.data.get("class_name", [])
        hands_idx = [i for i, n in enumerate(class_names) if n == "human hand"]
        utensils_idx = [i for i, n in enumerate(class_names) if n in UTENSIL_CLASSES]
        food_idx = [i for i, n in enumerate(class_names) if n != "human hand" and n not in UTENSIL_CLASSES]

        dets_hands = detections[np.array(hands_idx)] if hands_idx else sv.Detections.empty()
        dets_utensils = detections[np.array(utensils_idx)] if utensils_idx else sv.Detections.empty()
        dets_food = detections[np.array(food_idx)] if food_idx else sv.Detections.empty()

        # Tracking for food only
        try:
            dets_food = self.tracker.update_with_detections(dets_food)
        except Exception:
            logging.exception("Tracker update failed")

        # Interaction check
        interacting_food_ids = set()

        def check_interaction(actuator_dets, actuator_type: str):
            for a_box in actuator_dets.xyxy:
                for i, f_box in enumerate(dets_food.xyxy):
                    if calculate_iou(a_box, f_box) > INTERACTION_IOU_THRESHOLD:
                        tid = int(dets_food.tracker_id[i])
                        interacting_food_ids.add(tid)
                        # Visuals
                        ax, ay = int((a_box[0]+a_box[2])/2), int((a_box[1]+a_box[3])/2)
                        fx, fy = int((f_box[0]+f_box[2])/2), int((f_box[1]+f_box[3])/2)
                        color = (0, 0, 255) if actuator_type == "hand" else (255, 255, 0)
                        cv2.line(color_image, (ax, ay), (fx, fy), color, 3)
                        fname = dets_food.data["class_name"][i]
                        current_interaction_display["text"] = f"Action: {actuator_type} -> {fname}"
                        current_interaction_display["timer"] = time.time()

        if len(dets_hands) > 0:
            check_interaction(dets_hands, "hand")
        if len(dets_utensils) > 0:
            check_interaction(dets_utensils, "utensil")

        # Manage state and schedule segmentation
        ids_to_seg: List[int] = []
        boxes_to_seg: List[List[int]] = []

        for i, tid in enumerate(dets_food.tracker_id):
            tid = int(tid)
            label = dets_food.data["class_name"][i]

            # Initialize state if new
            if tid not in TRACKED_OBJECT_CACHE:
                TRACKED_OBJECT_CACHE[tid] = FoodState(label)
            st = TRACKED_OBJECT_CACHE[tid]
            st.last_seen = time.time()

            is_overlapped = (tid in interacting_food_ids)

            # Transition: start interaction
            if is_overlapped and not st.is_interacting:
                st.is_interacting = True
                st.waiting_for_settle = False
                st.pre_interaction_vol_ml = st.current_avg_vol_ml
                logging.info(f"Interaction STARTED on {label} (#{tid}). Locked Vol: {st.pre_interaction_vol_ml:.1f}mL")

            # Transition: end interaction -> start settling
            elif not is_overlapped and st.is_interacting:
                st.is_interacting = False
                st.waiting_for_settle = True
                st.settling_start_time = time.time()
                logging.info(f"Interaction ENDED on {label} (#{tid}). Waiting for settle...")

            # Settling check
            if st.waiting_for_settle:
                if time.time() - st.settling_start_time > SETTLING_TIME:
                    st.waiting_for_settle = False
                    density = get_density(st.label)
                    pre_g = st.pre_interaction_vol_ml * density
                    post_g = st.current_weight_g
                    delta_g = pre_g - post_g
                    if delta_g > MIN_BITE_GRAMS:
                        msg = f"BITE DETECTED: {delta_g:.1f}g of {st.label}"
                        logging.info(msg)
                        current_interaction_display["text"] = msg
                        current_interaction_display["timer"] = time.time()
                    else:
                        logging.debug(f"Interaction delta {delta_g:.1f}g too small, ignored.")

            # Segment only if not overlapped (and not necessarily during settling)
            if not is_overlapped and not st.waiting_for_settle:
                boxes_to_seg.append(list(map(int, dets_food.xyxy[i])))
                ids_to_seg.append(tid)

        # Queue segmentation
        if boxes_to_seg:
            try:
                self.seg_job_queue_put_nonblocking(color_image.copy(), np.array(boxes_to_seg), ids_to_seg)
            except Exception:
                pass

        # Process segmentation results
        try:
            while True:
                masks, t_ids = self.seg_result_queue.get_nowait()
                for idx, mask in enumerate(masks):
                    if idx >= len(t_ids):
                        continue
                    tid = int(t_ids[idx])
                    if mask is None or tid not in TRACKED_OBJECT_CACHE:
                        continue
                    # Ensure mask size matches depth
                    if mask.shape != depth_map.shape:
                        try:
                            mask = cv2.resize(mask.astype(np.uint8),
                                              (depth_map.shape[1], depth_map.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            continue
                    vol_ml = estimate_volume_from_mask(mask, depth_map, self.intrinsics)
                    st = TRACKED_OBJECT_CACHE[tid]
                    # Do not contaminate moving average during interaction/settling
                    if not st.is_interacting and not st.waiting_for_settle:
                        density = get_density(st.label)
                        st.update_volume(vol_ml, density)
                self.seg_result_queue.task_done()
        except queue.Empty:
            pass

        # Draw UI
        labels = []
        for tid in dets_food.tracker_id:
            tid = int(tid)
            if tid in TRACKED_OBJECT_CACHE:
                s = TRACKED_OBJECT_CACHE[tid]
                w = s.current_weight_g
                status = "EATING" if s.is_interacting else ("SETTLING" if s.waiting_for_settle else "READY")
                labels.append(f"#{tid} {s.label}: {w:.0f}g [{status}]")
            else:
                labels.append(f"#{tid} ...")

        try:
            annotated_image = self.box_annotator.annotate(color_image.copy(), dets_food)
            annotated_image = self.label_annotator.annotate(annotated_image, dets_food, labels=labels)
            annotated_image = self.box_annotator.annotate(annotated_image, dets_hands)
            annotated_image = self.box_annotator.annotate(annotated_image, dets_utensils)

            # Interaction text
            if time.time() - current_interaction_display["timer"] < 4.0:
                cv2.putText(annotated_image, current_interaction_display["text"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Food & Bite Tracking (ROS2)", annotated_image)
            cv2.waitKey(1)
        except Exception:
            logging.exception("Annotation/display failed")

        # Cleanup stale trackers
        if time.time() - self.last_cleanup > 10.0:
            stale = [tid for tid, v in TRACKED_OBJECT_CACHE.items()
                     if (time.time() - v.last_seen) > TRACK_EVICT_SECONDS]
            for tid in stale:
                logging.debug(f"Evicting stale tracker {tid}")
                TRACKED_OBJECT_CACHE.pop(tid, None)
            self.last_cleanup = time.time()

    def seg_job_queue_put_nonblocking(self, color_img: np.ndarray, boxes: np.ndarray, ids: List[int]):
        try:
            self.seg_job_queue.put_nowait((color_img, boxes, ids))
        except queue.Full:
            logging.debug("Segmentation queue full; skipping this cycle")

    def destroy_node(self):
        try:
            self.seg_worker_stop.set()
            while not self.seg_job_queue.empty():
                self.seg_job_queue.get_nowait()
                self.seg_job_queue.task_done()
            while not self.seg_result_queue.empty():
                self.seg_result_queue.get_nowait()
                self.seg_result_queue.task_done()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FoodVolumeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()