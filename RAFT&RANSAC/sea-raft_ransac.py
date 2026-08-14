import os
import sys
import argparse
from glob import glob
import re

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add SEA-RAFT path
sys.path.append('/home/noik/sea_raft/SEA-RAFT/core')

from raft import RAFT
from utils.utils import load_ckpt
from config.parser import parse_args

# RANSAC utilities
from motion_detection_utils import *
from constrained_ransac import *


DEVICE = 'cuda'


def load_kitti_sequence(sequence_dir):
    # Find image folder
    possible_image_folders = ['image_0', 'image_1', 'image_2', 'image_3']
    
    image_folder = None
    for folder in possible_image_folders:
        folder_path = os.path.join(sequence_dir, folder)
        if os.path.exists(folder_path):
            image_folder = folder_path
            print(f"Found image folder: {folder}")
            break
    
    if image_folder is None:
        if any(seq_dir in sequence_dir for seq_dir in possible_image_folders):
            image_folder = sequence_dir
        else:
            raise ValueError(f"No image folder found in {sequence_dir}")
    
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    # Sort files naturally
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]
    
    image_files.sort(key=natural_sort_key)
    
    return image_files, image_folder

# ============================================================
# SEA-RAFT OPTICAL FLOW
# ============================================================

def compute_sea_raft_flow(model, args, frame1, frame2):
    # Convert BGR to RGB
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 tensors
    image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()
    image2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()
    
    # Add batch dimension
    image1 = image1.unsqueeze(0).to(DEVICE)
    image2 = image2.unsqueeze(0).to(DEVICE)
    
    # Apply scaling
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    
    # Forward pass
    with torch.no_grad():
        output = model(img1, img2, iters=args.iters, test_mode=True)
        flow_tensor = output['flow'][-1]
    
    # Scale back down
    flow_down = F.interpolate(flow_tensor, 
                             scale_factor=0.5 ** args.scale, 
                             mode='bilinear', 
                             align_corners=False) * (0.5 ** args.scale)
    
    # Convert to numpy
    flow = flow_down[0].permute(1, 2, 0).cpu().numpy()
    
    return flow


# ============================================================
# FALLBACK MOTION DETECTION
# ============================================================

def simple_motion_detection(flow, c=2.0, min_mag=5.0):
    """Fallback when RANSAC fails"""
    u_mean = flow[:, :, 0].mean()
    v_mean = flow[:, :, 1].mean()
    
    compensated_x = flow[:, :, 0] - u_mean
    compensated_y = flow[:, :, 1] - v_mean
    
    mag, _ = cv2.cartToPolar(compensated_x, compensated_y)
    
    threshold = mag.mean() + c * mag.std()
    absolute_threshold = max(threshold, min_mag)
        
    motion_mask = np.uint8(mag > absolute_threshold) * 255
    
    return motion_mask


# ============================================================
# SEA-RAFT + RANSAC MOTION DETECTION
# ============================================================

def detect_motion_sea_raft(model, args, frame1, frame2, ransac_thresh=1.0, ransac_iters=30, c=1.0, min_mag=5.0):
    # compute Deep Optical Flow
    flow = compute_sea_raft_flow(model, args, frame1, frame2)

    # Use RANSAC to obtain H matrix
    h, w, _ = flow.shape

    # get points P and polynomial expansion X
    P, X = get_px(w, h)

    # get sample index
    index, n_ttl, n_s = get_sampling_index(w, h, s=25, p=0.5)

    # obtain H matrix
    H, e = cra_fast(flow, P, X, index, n_ttl, n_s, thresh=ransac_thresh, min_inliers=5000, num_iters=ransac_iters)

    if np.all(H == 0) or e > 1e9:
        return simple_motion_detection(flow, c=0.5, min_mag=5.0)
    
    # use H matrix to get estimated background and foreground
    Fb = (X @ H) - P
    background_flow = Fb.reshape(flow.shape)
    
    foreground_flow = flow - background_flow
    mag_f, _ = cv2.cartToPolar(foreground_flow[:, :, 0], foreground_flow[:, :, 1])
    
    # threshold foreground flow to get motion mask
    threshold = mag_f.mean() + c * mag_f.std(ddof=1)    
    absolute_threshold = max(threshold, min_mag)
    motion_mask = np.uint8(mag_f > absolute_threshold) * 255
    
    return motion_mask


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description='SEA-RAFT Optical Flow Motion Mask Generation')
    parser.add_argument('--data_dir',type=str,required=True,help='Path to KITTI sequence')
    parser.add_argument('--output_dir',type=str,default='raft_motion_masks',help='Directory to save motion masks')
    parser.add_argument('--cfg',type=str,required=True,help='SEA-RAFT configuration file')
    parser.add_argument('--model',type=str,required=True,help='SEA-RAFT checkpoint')
    parser.add_argument('--iters',type=int,default=12,help='Number of SEA-RAFT iterations')
    parser.add_argument('--num_frames', type=int, default=None,help='Number of frames to process (default: all)')
    args = parse_args(parser)

     # Load KITTI sequence data
    image_files, image_folder = load_kitti_sequence(args.data_dir)
    
    if args.num_frames:
        image_files = image_files[:args.num_frames + 1]  # +1 because we need pairs
    
    print(f"\n=== Processing {len(image_files)-1} frame pairs ===")
    
    # Load SEA-RAFT model
    raft_model = RAFT(args)
    load_ckpt(raft_model, args.model)
    raft_model.to(DEVICE)
    raft_model.eval()

    # Initialize
    prev_frame = cv2.imread(image_files[0])
    H, W, _ = prev_frame.shape
    H2, W2 = 324, 720  # Resize dimensions for RAFT

    # Process frames
    for i in range(1, len(image_files)):
        
        curr_frame = cv2.imread(image_files[i])
        
        prev_name = os.path.basename(image_files[i-1])
        curr_name = os.path.basename(image_files[i])
        
        print(f"\rProcessing pair {i}/{len(image_files)-1}: {prev_name} -> {curr_name}", end="")
        print(f"  Previous: {prev_name} (frame {i-1})")
        print(f"  Current:  {curr_name} (frame {i})")

        # ----------------------------------------------------
        # Resize frames for SEA-RAFT
        # ----------------------------------------------------

        prev_small = cv2.resize(prev_frame,(W2, H2))

        curr_small = cv2.resize(curr_frame,(W2, H2))

        # ----------------------------------------------------
        # SEA-RAFT + RANSAC
        # ----------------------------------------------------

        raft_mask = detect_motion_sea_raft(
            raft_model,
            args,
            prev_small,
            curr_small
        )

        # ----------------------------------------------------
        # Resize motion mask back to original resolution
        # ----------------------------------------------------

        raft_mask_full = cv2.resize(raft_mask,(W, H),interpolation=cv2.INTER_NEAREST)

        # ----------------------------------------------------
        # Save motion mask
        # ----------------------------------------------------

        base_name = os.path.splitext(prev_name)[0]

        output_path = os.path.join(args.output_dir,f"{base_name}_motion_mask.png")

        cv2.imwrite(output_path,raft_mask_full)

        print(f"Saved motion mask: {output_path}")

        # ----------------------------------------------------
        # Optional: save optical-flow visualization
        # ----------------------------------------------------
        # If you want this later, we can add it.
        
        # Next frame
        prev_frame = curr_frame.copy()

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()