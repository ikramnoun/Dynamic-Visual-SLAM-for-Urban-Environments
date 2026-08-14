import os
import sys
import argparse
from glob import glob
import re
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import time  # ADD THIS for timestamp
import torch.nn.functional as F

# ROS2 imports (minimal)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage  # CHANGED from Image to CompressedImage
from cv_bridge import CvBridge

# Add SEA-RAFT path and import components
sys.path.append('/home/noik/sea_raft/SEA-RAFT/core')
from raft import RAFT
from utils.utils import load_ckpt
from config.parser import parse_args

# Import your RANSAC utilities
from motion_detection_utils import *
from constrained_ransac import *

DEVICE = 'cuda'

def load_kitti_sequence(sequence_dir):

    # Find image folder (could be image_0, image_1, etc.)
    possible_image_folders = [
        'image_0',      # Left grayscale
        'image_1',      # Right grayscale  
        'image_2',      # Left color
        'image_3',      # Right color
    ]
    
    image_folder = None
    for folder in possible_image_folders:
        folder_path = os.path.join(sequence_dir, folder)
        if os.path.exists(folder_path):
            image_folder = folder_path
            print(f"Found image folder: {folder}")
            break
    
    if image_folder is None:
        # Check if sequence_dir already points to image folder
        if any(seq_dir in sequence_dir for seq_dir in possible_image_folders):
            image_folder = sequence_dir
        else:
            raise ValueError(f"No image folder found in {sequence_dir}")
    
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    # Sort files naturally
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]
    
    image_files.sort(key=natural_sort_key)
    
    # USE SYNTHETIC 10Hz TIMESTAMPS 
    timestamps = [i * 0.1 for i in range(len(image_files))]
    
    return image_files, timestamps, image_folder


class CompressedPublisher(Node):
    def __init__(self):
        super().__init__('compressed_publisher')
        self.bridge = CvBridge()
        
        # SINGLE publisher for combined data
        self.publisher = self.create_publisher(CompressedImage, 
                                              '/camera/compressed_with_mask', 
                                              10)
        
        self.get_logger().info('Compressed publisher ready')
    
    def publish_combined(self, image, mask, frame_number, timestamp):
        """
        Combine image and mask into single compressed message
        Format: [4-byte image_size][jpeg_image_data][png_mask_data]
        """
        # Validate inputs
        if image is None or mask is None:
            self.get_logger().warn(f"Skipping frame {frame_number} - None input")
            return
            
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # 1. Compress image as JPEG (fast, good compression)
        success_img, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success_img:
            self.get_logger().error(f"Failed to encode image for frame {frame_number}")
            return
            
        # 2. Compress mask as PNG (preserves binary data)
        success_mask, mask_encoded = cv2.imencode('.png', mask)
        if not success_mask:
            self.get_logger().error(f"Failed to encode mask for frame {frame_number}")
            return
        
        # 3. Create combined data
        img_size = len(img_encoded)
        
        # First 4 bytes: image size as little-endian
        size_bytes = img_size.to_bytes(4, 'little')
        
        # Combine: size + image + mask
        combined_data = size_bytes + img_encoded.tobytes() + mask_encoded.tobytes()
        
        # 4. Create ROS2 message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"frame_{frame_number:06d}"
        msg.format = "jpeg+png"  # Custom format identifier
        msg.data = combined_data
        
        # 5. Publish
        self.publisher.publish(msg)
        
        self.get_logger().info(f'Published frame {frame_number} at {timestamp:.6f}s')
        self.get_logger().debug(f'Image: {image.shape}, Mask: {mask.shape}, Combined: {len(combined_data)} bytes')

# Global dictionary to track object state history
object_state_history = defaultdict(lambda: {'dynamic_count': 0, 'static_count': 0})
object_recent_states = defaultdict(lambda: deque(maxlen=3))  # Track last 3 states

def compute_sea_raft_flow(model, args, frame1, frame2):
    """
    Correct flow computation matching demo.py
    """
    # Convert BGR to RGB
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 tensors (range 0-255)
    image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()
    image2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()
    
    # Add batch dimension
    image1 = image1.unsqueeze(0).to(DEVICE)
    image2 = image2.unsqueeze(0).to(DEVICE)
    
    # Apply scaling (from demo.py)
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    
    # Forward pass
    with torch.no_grad():
        output = model(img1, img2, iters=args.iters, test_mode=True)
        flow_tensor = output['flow'][-1]
    
    # Scale back down (CRITICAL - exactly as in demo)
    flow_down = F.interpolate(flow_tensor, 
                             scale_factor=0.5 ** args.scale, 
                             mode='bilinear', 
                             align_corners=False) * (0.5 ** args.scale)
    
    # Convert to numpy
    flow = flow_down[0].permute(1, 2, 0).cpu().numpy()
    
    return flow


def simple_motion_detection(flow, c=2.0, min_mag=5.0):
    """Fallback when RANSAC fails - simple global motion compensation"""
    h, w = flow.shape[:2]
    
    # Method 1: Subtract global mean flow
    u_mean = flow[:, :, 0].mean()
    v_mean = flow[:, :, 1].mean()
    
    compensated_x = flow[:, :, 0] - u_mean
    compensated_y = flow[:, :, 1] - v_mean
    
    # Compute magnitude of compensated flow
    mag, _ = cv2.cartToPolar(compensated_x, compensated_y)
    
    # Adaptive threshold
    threshold = mag.mean() + c * mag.std()
    absolute_threshold = max(threshold, min_mag)
        
    motion_mask = np.uint8(mag > absolute_threshold) * 255
    
    # Light cleanup
    #kernel = np.ones((3, 3), np.uint8)
    #motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    return motion_mask

def detect_motion_sea_raft(model, args,frame1, frame2, ransac_thresh=1.0, ransac_iters=30,  c=1.0, min_mag=5.0):
    
    # compute Deep Optical Flow
    flow = compute_sea_raft_flow(model, args, frame1, frame2)

    # Use RANSAC to obtain H matrix
    h, w, _ = flow.shape

    # get points P and polynomial expansion X
    P, X = get_px(w, h)

    # get sample index
    index, n_ttl, n_s = get_sampling_index(w, h, s=25, p=0.5)

    # obtain H matrix
    #H, e = cra(flow, P, X, index, n_ttl, n_s, thresh=ransac_thresh, min_inliers=10000, num_iters=ransac_iters)
    H, e = cra_fast(flow, P, X, index, n_ttl, n_s, thresh=ransac_thresh, min_inliers=5000, num_iters=ransac_iters)
    print('ransac_error:', e)

    if np.all(H == 0) or e > 1e9:
        print("  RANSAC failed, using fallback...")
        return simple_motion_detection(flow, c=0.5, min_mag=5.0)
    
    # use H matrix to get estimated background and foreground
    Fb = (X @ H) - P
    background_flow = Fb.reshape(flow.shape)
    
    foreground_flow = flow - background_flow
    mag_f, _ = cv2.cartToPolar(foreground_flow[:, :, 0], foreground_flow[:, :, 1])
    
    # threshold foreground flow to get motion mask
    threshold = mag_f.mean() + c*mag_f.std(ddof=1)    
    absolute_threshold = max(threshold, min_mag)
    motion_mask = np.uint8(mag_f > absolute_threshold) * 255

    # (OPTIONAL) clean motion mask
    #kernel = np.ones((3,3))
    # motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    #motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)  
    
    return motion_mask

def generate_mask(image_path):
    """Generate YOLO mask points and track IDs for an image - Most efficient version"""
    frame = cv2.imread(image_path)
    
    frame_shape = frame.shape[:2]  
    
    # Run tracking
    #results = yolo_model.track(frame, persist=True, classes=[0, 2, 5, 7])
    results = yolo_model.track(frame, persist=True, classes=[0, 2, 5, 7],conf=0.3,iou=0.5,tracker="bytetrack.yaml", verbose=False)
    
    # Store mask points and track IDs
    object_data = []  # List of (track_id, mask_points, bbox)
    combined_mask = np.zeros(frame_shape, dtype=np.uint8)
    
    if results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1]*len(boxes)
        masks = results[0].masks.xy
        
        for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
            if mask.size > 0:
                mask_points = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                
                # Store mask points, track ID, and bounding box
                object_data.append((track_id, mask_points, box))
                
                # Fill combined mask for visualization
                cv2.fillPoly(combined_mask, [mask_points], 255)
    
    return object_data, combined_mask

def check_points_intersection_fast(mask_points, raft_mask):

    points = mask_points.reshape(-1, 2)
    
    sampled_points = points[::1]
    total_sampled = len(sampled_points)
    
    intersecting_points = 0
    
    for point in sampled_points:
        x, y = int(point[0]), int(point[1])
        if (0 <= y < raft_mask.shape[0] and 
            0 <= x < raft_mask.shape[1] and 
            raft_mask[y, x] == 255):
            intersecting_points += 1
    
    return (intersecting_points / total_sampled) >= 0.1

def classify_objects_by_motion(object_data, raft_mask):
    """Classify objects as dynamic or static based on RAFT intersection with edge handling"""
    dynamic_objects = []

    for track_id, mask_points, bbox in object_data:
        
        # Check if this object's mask points intersect with RAFT mask
        intersects_raft = check_points_intersection_fast(mask_points, raft_mask)
        
        # Determine current state
        current_state = 'dynamic' if intersects_raft else 'static'

        if current_state == 'dynamic':
            dynamic_objects.append((track_id, mask_points, bbox))

    return dynamic_objects

def create_dynamic_mask(dynamic_objects, shape):
    """Create combined mask from dynamic objects using their original mask points"""
    dynamic_mask = np.zeros(shape, dtype=np.uint8)
    
    for track_id, mask_points, bbox in dynamic_objects:
        # Use the exact same mask points from YOLO
        cv2.fillPoly(dynamic_mask, [mask_points], 255)
    
    return dynamic_mask


def main():
    # Initialize ROS2
    rclpy.init()
    publisher = CompressedPublisher()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combined YOLO + RAFT Processing')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to KITTI sequence folder (e.g., 04/)')
    parser.add_argument('--output_dir', type=str, default='output_01',
                       help='Root output directory for all results')
    
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    parser.add_argument('--iters', type=int, default=12, help='number of iterations for RAFT')
    
    args = parse_args(parser) 

    # Load KITTI sequence data
    image_files, timestamps, image_folder = load_kitti_sequence(args.data_dir)

    # Create organized output directories under root folder
    root_dir = args.output_dir
    yolo_masks_dir = os.path.join(root_dir, 'yolo_masks')
    raft_masks_dir = os.path.join(root_dir, 'raft_masks')
    dynamic_masks_dir = os.path.join(root_dir, 'dynamic_masks')
    results_dir = os.path.join(root_dir, 'results')
    
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(yolo_masks_dir, exist_ok=True)
    os.makedirs(raft_masks_dir, exist_ok=True)
    os.makedirs(dynamic_masks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load SEA-RAFT model
    raft_model = RAFT(args)
    load_ckpt(raft_model, args.model)
    raft_model.to(DEVICE)
    raft_model.eval()

    # Initialize
    prev_frame = cv2.imread(image_files[0])
    H, W, _ = prev_frame.shape
    H2, W2 = 324, 720
    #H2, W2 = H//2, W//2

    # Initialize tracking lists
    all_dynamic_objects = []

    # Process frames
    for i in range(1, len(image_files)):
        curr_frame = cv2.imread(image_files[i])

        # Get frame names for debugging
        prev_name = os.path.basename(image_files[i-1])
        curr_name = os.path.basename(image_files[i])

        print(f"\n=== Processing pair {i}/{len(image_files)-1} ===")
        print(f"  Previous: {prev_name} (frame {i-1})")
        print(f"  Current:  {curr_name} (frame {i})")
        
        # === PARALLEL PROCESSING ===
        with ThreadPoolExecutor(max_workers=2) as executor:
            yolo_future = executor.submit(generate_mask, image_files[i-1])
            raft_future = executor.submit(
                detect_motion_sea_raft, 
                raft_model, args,
                cv2.resize(prev_frame, (W2, H2)),
                cv2.resize(curr_frame, (W2, H2))
            )
            
            object_data, yolo_combined_mask = yolo_future.result()
            raft_mask = raft_future.result()

        # Resize RAFT mask to full resolution
        raft_mask_full = cv2.resize(raft_mask, (W, H))
        yolo_combined_mask= cv2.resize(yolo_combined_mask, (W, H))
        
        # Classify objects as dynamic or static w
        dynamic_objects = classify_objects_by_motion(object_data, raft_mask_full  )
        
        # Extract track IDs for storage
        dynamic_track_ids = [track_id for track_id, _, _ in dynamic_objects]

        # Use the PREVIOUS frame name for output
        base_name = os.path.splitext(os.path.basename(image_files[i-1]))[0]

        # Store for final summary
        all_dynamic_objects.append({'frame': base_name, 'track_ids': dynamic_track_ids})
        
        # Create mask from dynamic objects using original YOLO mask points
        dynamic_mask = create_dynamic_mask(dynamic_objects, (H, W))
        #dynamic_mask= yolo_combined_mask
        
        # Apply morphological operations
        #kernel = np.ones((3, 3), np.uint8)
        #dynamic_mask = cv2.dilate(dynamic_mask, kernel, iterations=1)
        
        # Ensure mask is uint8
        if dynamic_mask.dtype != np.uint8:
            dynamic_mask = dynamic_mask.astype(np.uint8)
        
        # Apply mask to PREVIOUS frame and draw tracking IDs
        final_mask = cv2.bitwise_not(dynamic_mask)
        result = cv2.bitwise_and(prev_frame, prev_frame, mask=final_mask)
        
        # Save YOLO mask
        cv2.imwrite(os.path.join(yolo_masks_dir, f"{base_name}_yolo_mask.jpg"), yolo_combined_mask)
        
        # Save RAFT mask
        cv2.imwrite(os.path.join(raft_masks_dir, f"{base_name}_raft_mask.jpg"), raft_mask_full)

        # Save dynamic mask
        cv2.imwrite(os.path.join(dynamic_masks_dir, f"{base_name}_dynamic_mask.jpg"), dynamic_mask)
        
        # Save result image
        cv2.imwrite(os.path.join(results_dir, f"{base_name}_result.jpg"), result)
        
        # Publish to ROS2 with KITTI timestamp
        #publisher.publish_combined(prev_frame, dynamic_mask,i-1,timestamps[i-1])

        prev_frame = curr_frame.copy()
    
    # Cleanup ROS2
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    # Initialize YOLO model globally
    yolo_model = YOLO("/home/noik/Downloads/yolo11s-seg.pt")
    main()