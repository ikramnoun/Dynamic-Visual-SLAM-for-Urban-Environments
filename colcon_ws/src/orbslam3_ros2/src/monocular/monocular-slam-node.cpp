#include "monocular-slam-node.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/imgcodecs.hpp>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    
    // SINGLE subscriber for combined data
    m_compressed_subscriber = this->create_subscription<CompressedImage>(
        "/camera/compressed_with_mask",
        10,
        std::bind(&MonocularSlamNode::GrabCompressed, this, _1));
    
    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 node with compressed input ready");
}

MonocularSlamNode::~MonocularSlamNode()
{
    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabCompressed(const CompressedImage::SharedPtr msg)
{
    try
    {
        // 1. Extract image size from first 4 bytes
        if (msg->data.size() < 4) {
            RCLCPP_ERROR(this->get_logger(), "Message too short");
            return;
        }
        
        uint32_t img_size;
        memcpy(&img_size, msg->data.data(), 4);
        
        // 2. Validate sizes
        if (msg->data.size() < 4 + img_size) {
            RCLCPP_ERROR(this->get_logger(), "Invalid data size");
            return;
        }
        
        // 3. Extract image data (JPEG)
        std::vector<uchar> img_data(msg->data.begin() + 4, 
                                    msg->data.begin() + 4 + img_size);
        
        // 4. Extract mask data (PNG)
        std::vector<uchar> mask_data(msg->data.begin() + 4 + img_size, 
                                     msg->data.end());
        
        // 5. Decode image
        cv::Mat image = cv::imdecode(img_data, cv::IMREAD_COLOR);
        if (image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to decode image");
            return;
        }
        
        // 6. Decode mask
        cv::Mat mask = cv::imdecode(mask_data, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty mask, using regular tracking");
            double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
            m_SLAM->TrackMonocular(image, timestamp);
            return;
        }
        
        // 7. Get timestamp
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
        std::string frame_id = msg->header.frame_id;
        
        // 8. Resize mask if needed (should match image size)
        if (mask.size() != image.size()) {
            cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
            RCLCPP_DEBUG(this->get_logger(), "Resized mask to match image");
        }
        
        // 9. Process with ORB-SLAM3
        RCLCPP_INFO(this->get_logger(), 
                   "Processing %s with mask at time: %.6f", 
                   frame_id.c_str(), timestamp);
        
        m_SLAM->TrackMonocularWithMask(image, mask, timestamp);
        
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}
