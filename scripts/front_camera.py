#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np


class FrontCameraNode(Node):
    def __init__(self):
        super().__init__('front_camera')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # ROS Parameters
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0) 
        self.declare_parameter('roi_width', 400)
        self.declare_parameter('roi_height', 300)
        
        # Get parameters
        min_detection_confidence = self.get_parameter('min_detection_confidence').value
        min_tracking_confidence = self.get_parameter('min_tracking_confidence').value
        
        # Initialize MediaPipe Holistic with refined landmarks
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize Face Mesh with refined landmarks (includes iris)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris detection (478 landmarks)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # ROI state
        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_x = self.get_parameter('roi_x').value
        self.roi_y = self.get_parameter('roi_y').value
        self.roi_width = self.get_parameter('roi_width').value
        self.roi_height = self.get_parameter('roi_height').value
        
        # Mouse state for ROI selection
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.temp_roi = None
        
        # Setup OpenCV window
        self.setup_opencv_window()
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera_01/color/image_raw', self.image_callback, 10)
        
        # Publishers
        self.annotated_pub = self.create_publisher(
            Image, '/front_camera/annotated_image', 10)
        self.pose_landmarks_pub = self.create_publisher(
            Float32MultiArray, '/front_camera/pose_landmarks', 10)
        self.face_landmarks_pub = self.create_publisher(
            Float32MultiArray, '/front_camera/face_landmarks', 10)
        self.left_hand_landmarks_pub = self.create_publisher(
            Float32MultiArray, '/front_camera/left_hand_landmarks', 10)
        self.right_hand_landmarks_pub = self.create_publisher(
            Float32MultiArray, '/front_camera/right_hand_landmarks', 10)
        
        self.get_logger().info('Front Camera Node initialized with refined face mesh (478 landmarks + iris)')

    def setup_opencv_window(self):
        """Setup OpenCV window for ROI selection"""
        try:
            cv2.namedWindow('Front Camera - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Front Camera - ROI Selection', self.mouse_callback)
            self.get_logger().info('OpenCV window setup for ROI selection')
        except Exception as e:
            self.get_logger().error(f'Failed to setup OpenCV window: {str(e)}')

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection with drag visualization"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.get_logger().info(f'ROI start point: ({x}, {y})')
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.start_point:
                self.dragging = False
                self.end_point = (x, y)
                
                # Calculate ROI coordinates
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                self.roi_x = x1
                self.roi_y = y1
                self.roi_width = x2 - x1
                self.roi_height = y2 - y1
                self.roi_enabled = True
                
                # Update parameters
                self.set_parameters([
                    rclpy.Parameter('roi_enabled', value=True),
                    rclpy.Parameter('roi_x', value=self.roi_x),
                    rclpy.Parameter('roi_y', value=self.roi_y),
                    rclpy.Parameter('roi_width', value=self.roi_width),
                    rclpy.Parameter('roi_height', value=self.roi_height)
                ])
                
                self.get_logger().info(f'ROI set: x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height}')

    def process_image(self, cv_image):
        """Process image with MediaPipe Holistic and Face Mesh"""
        height, width = cv_image.shape[:2]
        
        # Apply ROI if enabled
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            # Ensure ROI is within image bounds
            roi_x = max(0, min(self.roi_x, width - 1))
            roi_y = max(0, min(self.roi_y, height - 1))
            roi_x2 = min(width, roi_x + self.roi_width)
            roi_y2 = min(height, roi_y + self.roi_height)
            
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe Holistic
        holistic_results = self.holistic.process(image_rgb)
        
        # Process with MediaPipe Face Mesh (refined with iris)
        face_results = self.face_mesh.process(image_rgb)
        
        # Make image writable for drawing
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw holistic landmarks
        if holistic_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                holistic_results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        if holistic_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                holistic_results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        
        if holistic_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                holistic_results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        
        # Draw refined face mesh with iris
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw contours
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Draw iris (if refined landmarks are available)
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        
        # Create full image with annotations
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            full_annotated_image = cv_image.copy()
            full_annotated_image[roi_y:roi_y2, roi_x:roi_x2] = annotated_image
        else:
            full_annotated_image = annotated_image
        
        # Extract landmarks data
        pose_landmarks = self.extract_pose_landmarks(holistic_results, width, height, roi_offset)
        face_landmarks = self.extract_face_landmarks(face_results, width, height, roi_offset)
        left_hand_landmarks = self.extract_hand_landmarks(holistic_results.left_hand_landmarks, width, height, roi_offset)
        right_hand_landmarks = self.extract_hand_landmarks(holistic_results.right_hand_landmarks, width, height, roi_offset)
        
        return full_annotated_image, pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks

    def extract_pose_landmarks(self, results, width, height, roi_offset):
        """Extract pose landmarks"""
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * width + roi_offset[0]
                y = landmark.y * height + roi_offset[1]
                z = landmark.z
                landmarks.extend([x, y, z])
        return landmarks

    def extract_face_landmarks(self, results, width, height, roi_offset):
        """Extract face landmarks (478 points including iris)"""
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = landmark.x * width + roi_offset[0]
                    y = landmark.y * height + roi_offset[1]
                    z = landmark.z
                    landmarks.extend([x, y, z])
        return landmarks

    def extract_hand_landmarks(self, hand_landmarks, width, height, roi_offset):
        """Extract hand landmarks"""
        landmarks = []
        if hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x * width + roi_offset[0]
                y = landmark.y * height + roi_offset[1]
                z = landmark.z
                landmarks.extend([x, y, z])
        return landmarks

    def image_callback(self, msg):
        """Image callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process image
            annotated_image, pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks = self.process_image(cv_image)
            
            # Display image with ROI selection
            display_image = annotated_image.copy()
            
            # Draw ROI rectangle if enabled
            if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
                cv2.rectangle(display_image, 
                            (self.roi_x, self.roi_y), 
                            (self.roi_x + self.roi_width, self.roi_y + self.roi_height), 
                            (0, 255, 0), 2)
                cv2.putText(display_image, 'ROI', 
                          (self.roi_x, self.roi_y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw temporary ROI during dragging
            if self.dragging and self.start_point and self.end_point:
                cv2.rectangle(display_image, self.start_point, self.end_point, (255, 0, 0), 2)
                cv2.putText(display_image, 'Selecting ROI...', 
                          (self.start_point[0], self.start_point[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add instructions
            cv2.putText(display_image, 'Drag to select ROI', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_image, f'Face landmarks: {len(face_landmarks)//3} points', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Front Camera - ROI Selection', display_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
            elif key == ord('r'):
                self.roi_enabled = False
                self.set_parameters([rclpy.Parameter('roi_enabled', value=False)])
                self.get_logger().info('ROI reset')
            
            # Publish results
            try:
                # Publish annotated image
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                
                # Publish landmarks
                pose_msg = Float32MultiArray()
                pose_msg.data = pose_landmarks
                self.pose_landmarks_pub.publish(pose_msg)
                
                face_msg = Float32MultiArray()
                face_msg.data = face_landmarks
                self.face_landmarks_pub.publish(face_msg)
                
                left_hand_msg = Float32MultiArray()
                left_hand_msg.data = left_hand_landmarks
                self.left_hand_landmarks_pub.publish(left_hand_msg)
                
                right_hand_msg = Float32MultiArray()
                right_hand_msg.data = right_hand_landmarks
                self.right_hand_landmarks_pub.publish(right_hand_msg)
                
            except Exception as e:
                self.get_logger().error(f'Error publishing data: {str(e)}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


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