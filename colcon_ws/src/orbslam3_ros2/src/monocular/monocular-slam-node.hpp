#ifndef __MONOCULAR_SLAM_NODE_HPP__
#define __MONOCULAR_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "System.h"

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);
    ~MonocularSlamNode();

private:
    using CompressedImage = sensor_msgs::msg::CompressedImage;
    
    void GrabCompressed(const CompressedImage::SharedPtr msg);

    ORB_SLAM3::System* m_SLAM;
    rclcpp::Subscription<CompressedImage>::SharedPtr m_compressed_subscriber;
};

#endif
