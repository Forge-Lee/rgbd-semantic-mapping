#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>


namespace rgbd_mapping {

using Vec3 = std::array<double, 3>;


struct VoxelKey {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    bool operator==(const VoxelKey& other) const {
        return (
            x == other.x &&
            y == other.y &&
            z == other.z
        );
    }
};


struct VoxelKeyHash {
    std::size_t operator()(
        const VoxelKey& key
    ) const;
};


struct SemanticVoxel {
    Vec3 point_sum{
        0.0,
        0.0,
        0.0
    };

    Vec3 color_sum{
        0.0,
        0.0,
        0.0
    };

    std::size_t count = 0;

    double confidence_sum = 0.0;

    std::unordered_map<int, double>
        label_scores;
};


struct SemanticVoxelOutput {
    VoxelKey key;

    Vec3 point;
    Vec3 rgb_color;

    int label;

    double semantic_agreement;
    double mean_model_confidence;

    std::size_t observation_count;
};


class SemanticVoxelMap {
public:
    explicit SemanticVoxelMap(
        double voxel_size
    );

    void update(
        const std::vector<Vec3>& points,
        const std::vector<Vec3>& rgb_colors,
        const std::vector<int>& labels,
        const std::vector<double>& confidences
    );

    std::vector<SemanticVoxelOutput>
    exportMap() const;

    std::size_t size() const;


private:
    double voxel_size_;

    std::unordered_map<
        VoxelKey,
        SemanticVoxel,
        VoxelKeyHash
    > voxels_;

    VoxelKey pointToVoxel(
        const Vec3& point
    ) const;
};


}  // namespace rgbd_mapping