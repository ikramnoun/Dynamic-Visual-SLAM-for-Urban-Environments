import os
import cv2
import glob
from tqdm import tqdm

def convert_raw_to_kitti_odometry_mono(raw_drive_path, output_seq_path):
    # Input folder for mono
    cam0_dir = os.path.join(raw_drive_path, "image_02", "data")
    timestamps_file = os.path.join(raw_drive_path, "image_02", "timestamps.txt")

    # Output folder
    image_0_out = os.path.join(output_seq_path, "image_0")
    os.makedirs(image_0_out, exist_ok=True)

    # Collect left images
    cam0_imgs = sorted(glob.glob(os.path.join(cam0_dir, "*.png")))
    assert len(cam0_imgs) > 0, "No images found in image_00/data!"
    
    # Replace timestamp calculation with synthetic 10Hz
    timestamps = [i * 0.1 for i in range(len(cam0_imgs))]

    print("Converting mono dataset...")
    for idx, img_path in enumerate(tqdm(cam0_imgs)):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(image_0_out, f"{idx:06d}.png"), img)

    # Write times.txt
    with open(os.path.join(output_seq_path, "times.txt"), "w") as f:
        for t in timestamps:
            f.write(f"{t:.6f}\n")

    print("Mono conversion completed.")
    print("Output directory:", output_seq_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert KITTI raw sync drive to odometry format (mono)")
    parser.add_argument("--raw_path", required=True, help="Path to KITTI raw drive (e.g., 2011_09_30_drive_0016_sync)")
    parser.add_argument("--seq", default="07", help="Odometry sequence number")
    parser.add_argument("--out_dir", default="./sequences", help="Output base directory")
    args = parser.parse_args()

    output_path = os.path.join(args.out_dir, args.seq)
    convert_raw_to_kitti_odometry_mono(args.raw_path, output_path)
