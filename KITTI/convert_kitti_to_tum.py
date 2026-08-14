#!/usr/bin/env python3
import numpy as np
import sys

def convert_kitti_to_tum(kitti_file, output_file):
    """
    Convert KITTI odometry ground truth to TUM format
    Handles both 12-value (3x4) and 16-value (4x4) formats
    """
    
    # Read all lines
    with open(kitti_file, 'r') as f:
        lines = f.readlines()
    
    # Parse poses
    poses = []
    for line in lines:
        values = line.strip().split()
        if len(values) == 0:
            continue
            
        # Convert to float
        values = [float(v) for v in values]
        
        # Check format: 12 values (3x4) or 16 values (4x4)
        if len(values) == 12:
            # 3x4 transformation matrix
            pose = np.array([
                [values[0], values[1], values[2], values[3]],
                [values[4], values[5], values[6], values[7]],
                [values[8], values[9], values[10], values[11]],
                [0, 0, 0, 1]
            ])
        elif len(values) == 16:
            # 4x4 transformation matrix
            pose = np.array(values).reshape(4, 4)
        else:
            print(f"Warning: Line has {len(values)} values, expected 12 or 16")
            continue
        
        poses.append(pose)
    
    print(f"Loaded {len(poses)} poses")
    
    # Create timestamps (10 Hz for KITTI)
    timestamps = [i * 0.1 for i in range(len(poses))]
    
    # Write to TUM format
    with open(output_file, 'w') as f:
        for i, (timestamp, pose) in enumerate(zip(timestamps, poses)):
            # Extract translation (last column)
            tx = pose[0, 3]
            ty = pose[1, 3]
            tz = pose[2, 3]
            
            # Extract rotation matrix
            R = pose[:3, :3]
            
            # Convert to quaternion
            # Note: This is a simplified conversion - might need adjustment
            # based on your coordinate system
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
            
            # Normalize quaternion
            norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            qw /= norm
            qx /= norm
            qy /= norm
            qz /= norm
            
            # KITTI to TUM coordinate transformation:
            # KITTI: x=right, y=down, z=forward
            # TUM: x=forward, y=left, z=up
            tum_x = tz       # forward
            tum_y = -tx      # right -> left
            tum_z = -ty      # down -> up
            
            # Write in TUM format
            f.write(f"{timestamp:.6f} {tum_x:.6f} {tum_y:.6f} {tum_z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"Saved TUM format to {output_file}")
    
    # Print first and last pose for verification
    if len(poses) > 0:
        print(f"\nFirst pose (timestamp {timestamps[0]:.1f}s):")
        print(f"  Position: ({tum_x:.2f}, {tum_y:.2f}, {tum_z:.2f})")
        print(f"  Quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_kitti_to_tum.py <kitti_file> <output_tum_file>")
        print("Example: python convert_kitti_to_tum.py 04.txt ground_truth.tum")
        sys.exit(1)
    
    convert_kitti_to_tum(sys.argv[1], sys.argv[2])