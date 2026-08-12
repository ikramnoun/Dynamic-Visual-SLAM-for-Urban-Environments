# Dynamic Visual SLAM for Urban Environments

This repository contains the implementation of my Master's thesis project, **Dynamic Visual SLAM for Urban Environments**.

The project proposes a hybrid semantic-motion framework for detecting and filtering dynamic objects in urban environments. The system combines **YOLOv11 instance segmentation**, **SEA-RAFT optical flow**, **RANSAC-based motion detection**, and **ORB-SLAM3** to improve visual SLAM performance in dynamic scenes.

## System Overview

The proposed pipeline consists of four main components:

1. **YOLOv11 Instance Segmentation**
   Detects and segments potentially dynamic objects using semantic information.

2. **SEA-RAFT + RANSAC Motion Detection**
   Estimates dense optical flow and uses RANSAC to distinguish independently moving regions from the static background.

3. **Semantic-Motion Fusion**
   Combines the semantic segmentation results with the motion mask to generate the final dynamic-object mask.

4. **ROS2 + ORB-SLAM3 Integration**
   The generated dynamic mask is transmitted together with the corresponding camera image to ORB-SLAM3, where features located inside dynamic regions are filtered before being used by the SLAM system.

---

## 1. YOLOv11 Instance Segmentation

The semantic segmentation component is based on **YOLOv11**.

For the experiments, **10 dynamic classes** were selected from the Cityscapes dataset.

### Dataset

The YOLO training dataset containing the selected 10 dynamic classes can be downloaded from:

* [Cityscapes – 10 Dynamic Classes](https://app.roboflow.com/noik/cityscapes-zz0ur-5euoa/3)

For the complete Cityscapes dataset containing all **34 classes**, see:

* [Cityscapes – 34 Classes](https://app.roboflow.com/noik/cityscapes-zz0ur-7q6ok/2)

### Training

The `yolo_train.ipynb` notebook contains the code used to train the YOLOv11 instance segmentation model.

The trained model produces instance-level semantic masks for objects belonging to the selected dynamic classes.

---

## 2. SEA-RAFT Optical Flow and RANSAC Motion Detection

The motion detection component uses **SEA-RAFT** to estimate dense optical flow between consecutive frames.

The estimated optical flow is then processed using **RANSAC** to estimate the dominant background motion. Regions whose motion is inconsistent with the estimated background motion are identified as potentially dynamic regions.

### RAFT

For an introduction to RAFT optical flow and its implementation, refer to the original repository:

* [RAFT – Princeton-VL](https://github.com/princeton-vl/RAFT)

### SEA-RAFT

The SEA-RAFT implementation and usage instructions are available here:

* [SEA-RAFT – Princeton-VL](https://github.com/princeton-vl/SEA-RAFT)

### Motion Detection Notebook

`flow_based_motion_detection.ipynb`

This notebook demonstrates how optical flow and RANSAC can be combined for motion detection.

It can also be used to experiment with and fine-tune the RANSAC parameters in order to obtain suitable motion masks for the target dataset and environment.

### SEA-RAFT + RANSAC Script

`sea-raft_ransac.py`

This script combines SEA-RAFT optical flow estimation with RANSAC-based motion detection and generates **binary motion masks**.

---

## 3. Semantic-Motion Fusion and ROS2 Publishing

The semantic segmentation and motion detection outputs are combined to generate the final dynamic-object mask.

The basic idea is to use:

* **YOLOv11** → semantic information
* **SEA-RAFT + RANSAC** → motion information
* **Fusion** → identify objects that are both semantically dynamic and independently moving

### Real-Time Processing

`Real_time_Fusion&publisher.py`

This script runs the complete pipeline in real time.

It performs the following operations:

1. Receives consecutive camera frames.
2. Runs YOLOv11 instance segmentation.
3. Estimates optical flow using SEA-RAFT.
4. Performs RANSAC-based motion detection.
5. Fuses the semantic and motion information.
6. Generates the final dynamic mask.
7. Publishes the corresponding image and dynamic mask through ROS2.
8. Sends the data to the ORB-SLAM3 ROS2 node.
9. Saves the generated masks from the individual models in separate directories.

To reduce synchronization and processing delays, the image and its corresponding dynamic mask are transmitted together using a compressed ROS2 image message.

### Offline Mask Publisher

`Publisher.py`

This script is intended for experiments where the dynamic masks have already been generated.

Instead of performing YOLO and SEA-RAFT processing in real time, it reads the pre-generated dynamic masks from a specified directory and publishes them to ORB-SLAM3.

This approach is useful for:

* Faster experiments
* Repeated SLAM evaluation
* Comparing different SLAM configurations using the same masks
* Avoiding the computational cost of running the perception pipeline repeatedly

---

## 4. ORB-SLAM3 Integration

The SLAM component is based on **ORB-SLAM3**.

For information about building and configuring ORB-SLAM3, refer to:

* [ORB-SLAM3 Stereo Fixed](https://github.com/zang09/ORB-SLAM3-STEREO-FIXED)

### Modifications to ORB-SLAM3

Modifications were made to `Frame.cc` and `Frame.h` to allow ORB-SLAM3 to receive and use the generated dynamic mask.

The modified system uses the mask to identify dynamic regions and **filter out ORB features located inside those regions** before they are used by the SLAM system.

This prevents features belonging to independently moving objects from contributing to camera pose estimation and map construction.

---

## 5. ROS2 Integration

The ROS2 integration is based on the following ORB-SLAM3 ROS2 implementation:

* [ORB-SLAM3 ROS2](https://github.com/zang09/ORB_SLAM3_ROS2)

Additional modifications were made in:

```text
colcon_ws/
└── src/
    └── ...
        ├── monocular-slam-node.cpp
        └── monocular-slam-node.hpp
```

The modified ROS2 node:

1. Receives the compressed image containing the camera image and corresponding dynamic mask.
2. Decompresses the received data.
3. Separates the image and dynamic mask.
4. Passes them to the modified ORB-SLAM3 implementation.
5. Allows ORB-SLAM3 to filter features located inside dynamic regions.

---

## 6. Evaluation

The proposed system was evaluated on the **KITTI Odometry Benchmark** to compare the performance of the baseline ORB-SLAM3 system with the modified ORB-SLAM3 system incorporating dynamic-object filtering.

### Dataset

The evaluation was performed using the following KITTI Odometry sequences:

* **Sequence 01**
* **Sequence 04**
* **Sequence 06**
* **Sequence 07**
* **Sequence 09**

The KITTI Odometry dataset and development tools can be accessed from the official KITTI website:

* [KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

To download individual sequences and obtain the required ground-truth data, refer to the **KITTI Odometry Development Kit**.

### Converting KITTI Raw Data to Odometry Format

If the KITTI **raw dataset** is downloaded instead of the preprocessed Odometry dataset, the raw data needs to be converted into the required Odometry format before running the evaluation.

The repository provides:

```text id="m3f7v1"
raw_to_odometry.py
```

This script can be used to convert the downloaded KITTI raw data into the format required by the evaluation pipeline.

---

### Generating SLAM Trajectories

For the evaluation, the **offline publisher** can be used to replay the pre-generated dynamic masks without running the complete perception pipeline in real time.

The offline publisher allows ORB-SLAM3 to process the KITTI sequences and generate the corresponding keyframe trajectory.

The resulting trajectory files, such as:

```text id="q7k2s4"
KeyFrameTrajectory.txt
KeyFrameTrajectory_modified.txt
KeyFrameTrajectory_baseline.txt
```

can then be compared against the KITTI ground-truth trajectory.

---

### Converting Ground Truth to TUM Format

The KITTI ground-truth poses need to be converted into **TUM trajectory format** before they can be evaluated using the `evo` trajectory evaluation package.

The repository provides:

```text id="d1p8xz"
convert_kitti_to_tum.py
```

Use this script to convert the KITTI ground-truth poses into a `.tum` file, for example:

```text id="r4c9yw"
ground_truth.tum
```

---

### Trajectory Evaluation with EVO

The trajectory evaluation is performed using the Python package **evo**.

The system can be evaluated using two commonly used SLAM metrics:

* **Absolute Trajectory Error (ATE)** — evaluates the global consistency of the estimated trajectory with respect to the ground truth.
* **Relative Pose Error (RPE)** — evaluates the local accuracy of the estimated motion over a specified distance or time interval.

#### ATE Evaluation

To evaluate the Absolute Trajectory Error:

```bash id="h8v2kn"
evo_ape tum ground_truth.tum KeyFrameTrajectory.txt \
    --align \
    --correct_scale \
    --plot \
    --plot_mode xy \
    --save_plot ate_plot.png \
    --verbose \
    --save_results results_ape.zip
```

This command:

* Compares the estimated trajectory with the ground truth.
* Aligns the estimated trajectory with the reference trajectory.
* Corrects the scale.
* Calculates the ATE.
* Generates an XY trajectory plot.
* Saves the evaluation results to `results_ape.zip`.
* Saves the trajectory visualization as `ate_plot.png`.

---

#### RPE Evaluation

To evaluate the Relative Pose Error:

```bash id="v6s1qr"
evo_rpe tum ground_truth.tum KeyFrameTrajectory.txt \
    --align \
    --correct_scale \
    --delta 1 \
    --delta_unit m \
    --plot \
    --plot_mode xy \
    --save_plot rpe_plot.png \
    --save_results results_rpe.zip
```

Here, `--delta 1 --delta_unit m` evaluates the relative pose error over a **1-meter interval**.

The generated files include:

```text id="x2k9mc"
rpe_plot.png
results_rpe.zip
```

---

### Comparing Baseline and Modified ORB-SLAM3

To directly compare the trajectories produced by the baseline ORB-SLAM3 and the modified dynamic-object-aware ORB-SLAM3, use `evo_traj`:

```bash id="p5z8jd"
evo_traj tum \
    ground_truth.tum \
    KeyFrameTrajectory_modified.txt \
    KeyFrameTrajectory_baseline.txt \
    --ref=ground_truth.tum \
    --plot \
    --plot_mode xyz \
    --align \
    --correct_scale \
    --save_plot three_trajectories.png
```

This produces a visualization containing:

1. KITTI ground-truth trajectory
2. Baseline ORB-SLAM3 trajectory
3. Modified ORB-SLAM3 trajectory

The resulting plot can be used to visually analyze how dynamic-object filtering affects the estimated camera trajectory.

The evaluation results can then be used to quantitatively compare the baseline and proposed systems and determine whether filtering dynamic features improves SLAM trajectory accuracy.


## Acknowledgements

This project was developed as part of my Master's thesis in **Artificial Intelligence**.

The implementation builds upon several excellent open-source projects, including YOLO, SEA-RAFT, RAFT, ROS2, and ORB-SLAM3. Please refer to their respective repositories and publications for the original implementations and methodologies.
