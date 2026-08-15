#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include <vector>
#include <algorithm>
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
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"
#include "rgbd_mapping/semantic_voxel_map.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

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

std::array<uint8_t, 3> semanticColor(int label)
{
    // +1 avoids making label 0 pure black.
    uint32_t value =
        static_cast<uint32_t>(label + 1);

    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;

    for (int i = 0; i < 8; ++i) {
        r |= ((value >> 0) & 1) << (7 - i);
        g |= ((value >> 1) & 1) << (7 - i);
        b |= ((value >> 2) & 1) << (7 - i);

        value >>= 3;
    }

    return {r, g, b};
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

        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/semantic_mapping/map",
            10
        );

    }

private:
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
        const Image::ConstSharedPtr & confidence
    ){
        RCLCPP_INFO(
            this->get_logger(),
            "Entered processObservation"
        );

        const int stride = 4;

        // validate
        if (!validateObservation(rgb, depth, labels, confidence)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Observation validation failed | "
                "rgb=%s (%ux%u), "
                "depth=%s (%ux%u), "
                "labels=%s (%ux%u), "
                "confidence=%s (%ux%u)",
                rgb->encoding.c_str(), rgb->width, rgb->height,
                depth->encoding.c_str(), depth->width, depth->height,
                labels->encoding.c_str(), labels->width, labels->height,
                confidence->encoding.c_str(),
                confidence->width, confidence->height
            );
            return;
        }
        RCLCPP_INFO(
            this->get_logger(),
            "Observation validation passed"
        );

        // parse
        auto rgb_cv = cv_bridge::toCvShare(
            rgb,
            sensor_msgs::image_encodings::BGR8
        );

        auto depth_cv = cv_bridge::toCvShare(
            depth,
            sensor_msgs::image_encodings::TYPE_16UC1
        );

        auto labels_cv = cv_bridge::toCvShare(
            labels,
            sensor_msgs::image_encodings::TYPE_8UC1
        );

        auto confidence_cv = cv_bridge::toCvShare(
            confidence,
            sensor_msgs::image_encodings::TYPE_32FC1
        );

        const cv::Mat & rgb_img = rgb_cv->image;
        const cv::Mat & depth_img = depth_cv->image;
        const cv::Mat & labels_img = labels_cv->image;
        const cv::Mat & confidence_img = confidence_cv->image;

        const double fx = 525.0;
        const double fy = 525.0;
        const double cx = 319.5;
        const double cy = 239.5;

        const int height = depth_img.rows;
        const int width = depth_img.cols;

        std::vector<Vec3> points_world;
        std::vector<Vec3> rgb_colors;
        std::vector<int> semantic_labels;
        std::vector<double> confidences;

        for (int v = 0; v < height; v += stride) {
            for (int u = 0; u < width; u += stride) {

                uint16_t raw_depth = depth_img.at<uint16_t>(v, u);
                double depth_meter = decodeDepth(raw_depth);
                if (depth_meter < 0.1 || depth_meter > 5.0){
                    continue;
                }

                cv::Vec3b bgr = rgb_img.at<cv::Vec3b>(v, u);
                Vec3 rgb_color{
                    static_cast<double>(bgr[2]) / 255.0,
                    static_cast<double>(bgr[1]) / 255.0,
                    static_cast<double>(bgr[0]) / 255.0
                };

                uint8_t label = labels_img.at<uint8_t>(v, u);

                float conf = confidence_img.at<float>(v, u);

                // backproject
                Vec3 point = backprojectPixel(
                    u,
                    v,
                    depth_meter,
                    fx,
                    fy, 
                    cx,
                    cy
                );

                // transform
                Vec3 point_world = transformPointToWorld(point, *pose);
                
                points_world.push_back(point_world);
                rgb_colors.push_back(rgb_color);
                semantic_labels.push_back(static_cast<int>(label));
                confidences.push_back(static_cast<double>(conf));
               
            }
        }
        // map update
        voxel_map_.update(
            points_world,
            rgb_colors,
            semantic_labels,
            confidences
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Map update completed: %zu observations, %zu voxels",
            points_world.size(),
            voxel_map_.size()
        );

        publishMap(pose);
    }

    void publishMap(const PoseStamped::ConstSharedPtr & pose){
        const auto voxels =
            voxel_map_.exportMap();

        if (voxels.empty()) {
            return;
        }

        sensor_msgs::msg::PointCloud2 cloud;

        cloud.header.stamp = pose->header.stamp;

        cloud.header.frame_id = "world";

        cloud.height = 1;
        cloud.is_dense = true;

        sensor_msgs::PointCloud2Modifier modifier(
            cloud
        );

        modifier.setPointCloud2FieldsByString(
            2,
            "xyz",
            "rgb"
        );

        modifier.resize(
            voxels.size()
        );

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");

        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");

        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud, "r");

        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud, "g");

        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud, "b");

        for (const auto & voxel : voxels) {

            *iter_x = static_cast<float>(voxel.point[0]);
            *iter_y = static_cast<float>(voxel.point[1]);
            *iter_z = static_cast<float>(voxel.point[2]);

            const auto semantic_color = semanticColor(voxel.label);

            *iter_r = semantic_color[0];
            *iter_g = semantic_color[1];
            *iter_b = semantic_color[2];

            // *iter_r = static_cast<uint8_t>(
            //         std::clamp(
            //             voxel.rgb_color[0],
            //             0.0,
            //             1.0
            //         ) * 255.0
            //     );
            // *iter_g = static_cast<uint8_t>(
            //         std::clamp(
            //             voxel.rgb_color[1],
            //             0.0,
            //             1.0
            //         ) * 255.0
            //     );
            // *iter_b = static_cast<uint8_t>(
            //         std::clamp(
            //             voxel.rgb_color[2],
            //             0.0,
            //             1.0
            //         ) * 255.0
            //     );

            ++iter_x;
            ++iter_y;
            ++iter_z;
            ++iter_r;
            ++iter_g;
            ++iter_b;
        }

        map_pub_->publish(cloud);

        RCLCPP_INFO(
            this->get_logger(),
            "Published map with %zu voxels",
            voxels.size()
        );
    }

    rgbd_mapping::SemanticVoxelMap voxel_map_{0.02};

    message_filters::Subscriber<Image> rgb_sub_;
    message_filters::Subscriber<Image> depth_sub_;
    message_filters::Subscriber<PoseStamped> pose_sub_;
    message_filters::Subscriber<Image> labels_sub_;
    message_filters::Subscriber<Image> confidence_sub_;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
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