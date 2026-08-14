#!/usr/bin/env python3
import os
import argparse
from glob import glob
import re
import cv2
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

def load_kitti_sequence(sequence_dir):
    """Load KITTI sequence images and generate 10Hz timestamps"""
    # Find image folder
    possible_folders = ['image_0', 'image_1', 'image_2', 'image_3']
    image_folder = None
    for folder in possible_folders:
        folder_path = os.path.join(sequence_dir, folder)
        if os.path.exists(folder_path):
            image_folder = folder_path
            print(f"Found image folder: {folder}")
            break
    
    if image_folder is None:
        raise ValueError(f"No image folder found in {sequence_dir}")
    
    # Get image files
    image_files = sorted(glob(os.path.join(image_folder, '*.png')))
    if not image_files:
        image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    # Natural sort
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]
    image_files.sort(key=natural_sort_key)
    
    # 10Hz timestamps
    timestamps = [i * 0.1 for i in range(len(image_files))]
    
    return image_files, timestamps, image_folder


def load_mask_files(mask_dir):
    """Load pre-computed mask files"""
    mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
    if not mask_files:
        mask_files = sorted(glob(os.path.join(mask_dir, '*.jpg')))
    
    if not mask_files:
        raise ValueError(f"No mask files found in {mask_dir}")
    
    return mask_files


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(CompressedImage, '/camera/compressed_with_mask', 10)
    
    def publish(self, image, mask, frame_num, timestamp, image_name, mask_name=None):
        # Encode image as JPEG
        _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Encode mask as PNG
        _, mask_encoded = cv2.imencode('.png', mask)
        
        # Combine: [4-byte size][image data][mask data]
        img_size = len(img_encoded)
        combined = img_size.to_bytes(4, 'little') + img_encoded.tobytes() + mask_encoded.tobytes()
        
        # Create and publish message
        msg = CompressedImage()
        msg.header.stamp.sec = int(timestamp)
        msg.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        msg.header.frame_id = f"frame_{frame_num:06d}"
        msg.format = "jpeg+png"
        msg.data = combined
        
        self.publisher.publish(msg)
        
        # Print with image and mask names
        if mask_name:
            print(f"Published frame {frame_num}: image={image_name}, mask={mask_name} at {timestamp:.1f}s")
        else:
            print(f"Published frame {frame_num}: image={image_name} (no mask) at {timestamp:.1f}s")


def main():
    rclpy.init()
    publisher = MinimalPublisher()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='KITTI sequence folder')
    parser.add_argument('--mask_dir', required=True, help='Folder with pre-computed masks')
    parser.add_argument('--fps', type=float, default=10.0, help='Publishing rate')
    parser.add_argument('--no_mask', action='store_true', help='Publish without masks')
    args = parser.parse_args()
    
    # Load data
    image_files, timestamps, image_folder = load_kitti_sequence(args.data_dir)
    
    if not args.no_mask:
        mask_files = load_mask_files(args.mask_dir)
        # Ensure matching counts
        min_frames = min(len(image_files), len(mask_files))
        image_files = image_files[:min_frames]
        mask_files = mask_files[:min_frames]
        timestamps = timestamps[:min_frames]
        n_frames = min_frames
    else:
        mask_files = [None] * len(image_files)
        n_frames = len(image_files)
    
    # Print configuration
    print(f"\n=== Publisher Configuration ===")
    print(f"Images folder: {image_folder}")
    print(f"Images: {len(image_files)} frames")
    if not args.no_mask:
        print(f"Masks folder: {args.mask_dir}")
        print(f"Masks: {len(mask_files)} frames")
    print(f"Time range: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
    print(f"Publishing rate: {args.fps} Hz")
    print(f"Mode: {'Baseline (no mask)' if args.no_mask else 'With dynamic masks'}")
    print("=" * 30)
    print()  # Empty line for readability
    
    interval = 1.0 / args.fps
    
    try:
        for i in range(n_frames):
            # Get image filename only (not full path)
            image_name = os.path.basename(image_files[i])
            
            # Read image
            image = cv2.imread(image_files[i])
            if image is None:
                print(f"Warning: Could not read image {image_name}, skipping")
                continue
            
            if args.no_mask:
                # Create empty mask when no_mask is True
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                publisher.publish(image, mask, i, timestamps[i], image_name)
            else:
                # Get mask filename only
                mask_name = os.path.basename(mask_files[i])
                
                # Read mask
                mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not read mask {mask_name}, using empty mask")
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                publisher.publish(image, mask, i, timestamps[i], image_name, mask_name)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nPublisher stopped by user")
    
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import numpy as np
    main()