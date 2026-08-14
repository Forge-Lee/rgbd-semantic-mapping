#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Vector3.h"

using Image = sensor_msgs::msg::Image;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using Vec3 = std::array<double, 3>;

using SyncPolicy = message_filters::sync_policies::ExactTime<
        Image,
        Image,
        PoseStamped,
        Image,
        Image
    >;

bool validateObservation(
    const Image::ConstSharedPtr & rgb,
    const Image::ConstSharedPtr & depth,
    const Image::ConstSharedPtr & labels,
    const Image::ConstSharedPtr & confidence)
{
    if (rgb->width != depth->width ||
        rgb->height != depth->height) {
        return false;
    }

    if (depth->width != labels->width ||
        depth->height != labels->height) {
        return false;
    }

    if (labels->width != confidence->width ||
        labels->height != confidence->height) {
        return false;
    }

    if (rgb->encoding != "bgr8") {
        return false;
    }

    if (depth->encoding != "16UC1") {
        return false;
    }

    if (labels->encoding != "8UC1") {
        return false;
    }

    if (confidence->encoding != "32FC1") {
        return false;
    }

    return true;
}

double decodeDepth(uint16_t raw_depth){
    if (raw_depth == 0){
        return 0.0;
    }
    return raw_depth / 5000.0;
}

Vec3 backprojectPixel(
    int u,
    int v,
    double depth,
    double fx,
    double fy,
    double cx,
    double cy
){
    double z = depth;
    double x = (u - cx) * z / fx;
    double y = (v - cy) * z / fy;
    return Vec3{x, y, z};    
}

// tf2::Matrix3x3 getRotationMatrix(const PoseStamped & pose){
//     tf2::Quaternion q;

//     tf2::fromMsg(
//         pose.pose.orientation,
//         q
//     );

//     q.normalize();

//     tf2::Matrix3x3 R(q);

//     return R;
// }

Vec3 transformPointToWorld(
    const Vec3 & point_camera,
    const PoseStamped & pose
){
    // std::array<double, 4> points_homogeneous;
    // for (int i = 0; i < 3; i++){
    //     points_homogeneous[i] = point_camera[i];
    // } points_homogeneous[3] = 1;
    // tf2::Matrix3x3 R = getRotationMatrix(pose);
    tf2::Quaternion q;
    tf2::fromMsg(
        pose.pose.orientation,
        q
    );

    q.normalize();

    tf2::Matrix3x3 R(q);

    tf2::Vector3 p_camera(
        point_camera[0],
        point_camera[1],
        point_camera[2]
    );

    tf2::Vector3 t_world_camera(
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z
    );

    tf2::Vector3 p_world =
        R * p_camera + t_world_camera;

    return {
        p_world.x(),
        p_world.y(),
        p_world.z()
    };

}

class SemanticMappingNode : public rclcpp::Node
{
public:
    SemanticMappingNode(): Node("semantic_mapping_node"){
        rgb_sub_.subscribe(
            this,
            "/rgbd_replay/rgb"
        );

        depth_sub_.subscribe(
            this,
            "/rgbd_replay/depth"
        );

        pose_sub_.subscribe(
            this,
            "/rgbd_replay/pose"
        );

        labels_sub_.subscribe(
            this,
            "/semantic_segmentation/labels"
        );

        confidence_sub_.subscribe(
            this,
            "/semantic_segmentation/confidence"
        );

        synchronizer_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                SyncPolicy(20),
                rgb_sub_,
                depth_sub_,
                pose_sub_,
                labels_sub_,
                confidence_sub_
            );

        synchronizer_->registerCallback(
            std::bind(
                &SemanticMappingNode::synchronizedCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5
            )
        );

        // labels_subscription_ =
        //     this->create_subscription<sensor_msgs::msg::Image>(
        //         "/semantic_segmentation/labels",
        //         10,
        //         std::bind(
        //             &SemanticMappingNode::labelsCallback,
        //             this,
        //             std::placeholders::_1
        //         )
        //     );
        
        // confidence_subscription_ =
        //     this->create_subscription<sensor_msgs::msg::Image>(
        //         "/semantic_segmentation/confidence",
        //         10,
        //         std::bind(
        //             &SemanticMappingNode::confidenceCallback,
        //             this,
        //             std::placeholders::_1
        //         )
        //     );
        
        // rgb_subscription_ =
        //     this->create_subscription<sensor_msgs::msg::Image>(
        //         "/rgbd_replay/rgb",
        //         10,
        //         std::bind(
        //             &SemanticMappingNode::rgbCallback,
        //             this,
        //             std::placeholders::_1
        //         )
        //     );
        
        // depth_subscription_ =
        //     this->create_subscription<sensor_msgs::msg::Image>(
        //         "/rgbd_replay/depth",
        //         10,
        //         std::bind(
        //             &SemanticMappingNode::depthCallback,
        //             this,
        //             std::placeholders::_1
        //         )
        //     );

        // pose_subscription_ =
        //     this->create_subscription<geometry_msgs::msg::PoseStamped>(
        //         "/rgbd_replay/pose",
        //         10,
        //         std::bind(
        //             &SemanticMappingNode::poseCallback,
        //             this,
        //             std::placeholders::_1
        //         )
        //     );
    }

private:
    // void labelsCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    //     RCLCPP_INFO(
    //         this->get_logger(),
    //         "Received labels: shape=(%u, %u), encoding=%s, stamp=%d.%u",
    //         msg->height,
    //         msg->width,
    //         msg->encoding.c_str(),
    //         msg->header.stamp.sec,
    //         msg->header.stamp.nanosec
    //     );
    // }

    // void confidenceCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    //     RCLCPP_INFO(
    //         this->get_logger(),
    //         "Received confidence: shape=(%u, %u), encoding=%s, stamp=%d.%u",
    //         msg->height,
    //         msg->width,
    //         msg->encoding.c_str(),
    //         msg->header.stamp.sec,
    //         msg->header.stamp.nanosec
    //     );
    // }

    // void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    //     RCLCPP_INFO(
    //         this->get_logger(),
    //         "Received rgb: shape=(%u, %u), encoding=%s, stamp=%d.%u",
    //         msg->height,
    //         msg->width,
    //         msg->encoding.c_str(),
    //         msg->header.stamp.sec,
    //         msg->header.stamp.nanosec
    //     );
    // }

    // void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    //     RCLCPP_INFO(
    //         this->get_logger(),
    //         "Received depth: shape=(%u, %u), encoding=%s, stamp=%d.%u",
    //         msg->height,
    //         msg->width,
    //         msg->encoding.c_str(),
    //         msg->header.stamp.sec,
    //         msg->header.stamp.nanosec
    //     );
    // }

    // void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg){
    //     RCLCPP_INFO(
    //         this->get_logger(),
    //         "Received pose: position=(%.3f, %.3f, %.3f), "
    //         "quaternion=(%.3f, %.3f, %.3f, %.3f), "
    //         "stamp=%d.%u",
    //         msg->pose.position.x,
    //         msg->pose.position.y,
    //         msg->pose.position.z,
    //         msg->pose.orientation.x,
    //         msg->pose.orientation.y,
    //         msg->pose.orientation.z,
    //         msg->pose.orientation.w,
    //         msg->header.stamp.sec,
    //         msg->header.stamp.nanosec
    //     );
    // }

    void synchronizedCallback(
        const Image::ConstSharedPtr & rgb,
        const Image::ConstSharedPtr & depth,
        const PoseStamped::ConstSharedPtr & pose,
        const Image::ConstSharedPtr & labels,
        const Image::ConstSharedPtr & confidence)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "Synchronized observation | "
            "RGB=%d.%09u "
            "Depth=%d.%09u "
            "Pose=%d.%09u "
            "Labels=%d.%09u "
            "Confidence=%d.%09u",
            rgb->header.stamp.sec,
            rgb->header.stamp.nanosec,
            depth->header.stamp.sec,
            depth->header.stamp.nanosec,
            pose->header.stamp.sec,
            pose->header.stamp.nanosec,
            labels->header.stamp.sec,
            labels->header.stamp.nanosec,
            confidence->header.stamp.sec,
            confidence->header.stamp.nanosec
        );

        processObservation(
            rgb,
            depth,
            pose,
            labels,
            confidence
        );
    }

    void processObservation(
        const Image::ConstSharedPtr & rgb,
        const Image::ConstSharedPtr & depth,
        const PoseStamped::ConstSharedPtr & pose,
        const Image::ConstSharedPtr & labels,
        const Image::ConstSharedPtr & confidence){
        // validate
        if (!validateObservation(rgb, depth, labels, confidence)) {
            return;
        }
        // parse

        // backproject
        // transform
        // map update
    }

    message_filters::Subscriber<Image> rgb_sub_;
    message_filters::Subscriber<Image> depth_sub_;
    message_filters::Subscriber<PoseStamped> pose_sub_;
    message_filters::Subscriber<Image> labels_sub_;
    message_filters::Subscriber<Image> confidence_sub_;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
};


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node =
        std::make_shared<SemanticMappingNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}