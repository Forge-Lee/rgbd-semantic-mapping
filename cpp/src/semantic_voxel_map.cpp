#include "rgbd_mapping/semantic_voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace rgbd_mapping {


std::size_t VoxelKeyHash::operator()(
    const VoxelKey& key
) const {

    const std::size_t hx =
        std::hash<std::int64_t>{}(key.x);

    const std::size_t hy =
        std::hash<std::int64_t>{}(key.y);

    const std::size_t hz =
        std::hash<std::int64_t>{}(key.z);

    return (
        hx
        ^ (hy << 1)
        ^ (hz << 2)
    );
}


SemanticVoxelMap::SemanticVoxelMap(
    double voxel_size
)
    : voxel_size_(voxel_size)
{
    if (
        !std::isfinite(voxel_size_) ||
        voxel_size_ <= 0.0
    ) {
        throw std::invalid_argument(
            "voxel_size must be positive and finite"
        );
    }
}


VoxelKey SemanticVoxelMap::pointToVoxel(
    const Vec3& point
) const {
    // TODO 1
    //
    // 对照 Python:
    //
    Vec3 voxel_indices;
    for (int i = 0; i < 3; i++){
        voxel_indices[i] = std::floor(point[i] / voxel_size_);
    } return VoxelKey{
        static_cast<int64_t>(voxel_indices[0]),
        static_cast<int64_t>(voxel_indices[1]),
        static_cast<int64_t>(voxel_indices[2])
    };
    //
    // 注意负坐标！
    //
    // 返回:
    // VoxelKey{..., ..., ...}

    // throw std::runtime_error(
    //     "TODO: pointToVoxel"
    // );
}


void SemanticVoxelMap::update(
    const std::vector<Vec3>& points,
    const std::vector<Vec3>& rgb_colors,
    const std::vector<int>& labels,
    const std::vector<double>& confidences
) {
    // TODO 2:
    // sanity checks
    int set_size = points.size();
    if (set_size != rgb_colors.size()){
        throw std::runtime_error(
            "Different size between points and rgb colors"
        );
    }; if (set_size != labels.size()){
        throw std::runtime_error(
            "Different size between points and labels"
        );
    }; if (set_size != confidences.size()){
        throw std::runtime_error(
            "Different size between points and confidences"
        );
    };

    // TODO 3:
    // empty observation return
    if (points.size() == 0){
        return;
    }

    // TODO 4:
    // 遍历所有 observation:
    for (int i = 0; i < set_size; i++){
        // 1. point -> voxel key
        Vec3 curr_point = points[i];
        Vec3 curr_color = rgb_colors[i];
        double curr_confidence = confidences[i];
        int curr_label = labels[i];
        VoxelKey curr_key = pointToVoxel(curr_point);

        // 2. 找到 / 创建 accumulator
        if (voxels_.find(curr_key) == voxels_.end()){
            // key doesn't exist
            voxels_[curr_key];
        }; SemanticVoxel& accumulator = voxels_[curr_key];

        // 3. point_sum += point
        for (int j = 0; j < 3; j++){
            accumulator.point_sum[j] += curr_point[j];
        };

        // 4. color_sum += color
        for (int k = 0; k < 3; k++){
            accumulator.color_sum[k] += curr_color[k];
        };

        // 5. count += 1
        accumulator.count += 1;

        // 6. confidence_sum += confidence
        accumulator.confidence_sum += curr_confidence;

        // 7. label_scores[label] += confidence
        if (accumulator.label_scores.find(curr_label) == accumulator.label_scores.end()){
            accumulator.label_scores[curr_label] = curr_confidence;
        } else{
            accumulator.label_scores[curr_label] += curr_confidence;
        };
    }

}


std::vector<SemanticVoxelOutput>
SemanticVoxelMap::exportMap() const {
    // 先暂时不实现。
    //
    // 我们第一轮只验证 update 是否能正确
    // 创建和累计 voxel。

    return {};
}


std::size_t SemanticVoxelMap::size() const {
    // TODO 5

    return 0;
}


}  // namespace rgbd_mapping