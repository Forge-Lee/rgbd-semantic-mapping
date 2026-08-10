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
    Vec3 voxel_indices;
    for (int i = 0; i < 3; i++){
        voxel_indices[i] = std::floor(point[i] / voxel_size_);
    } return VoxelKey{
        static_cast<int64_t>(voxel_indices[0]),
        static_cast<int64_t>(voxel_indices[1]),
        static_cast<int64_t>(voxel_indices[2])
    };

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
    const std::size_t set_size = points.size();
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
    if (points.empty()){
        return;
    }

    // TODO 4:
    // 遍历所有 observation:
    for (int i = 0; i < set_size; i++){
        // 1. point -> voxel key
        const Vec3& curr_point = points[i];
        const Vec3& curr_color = rgb_colors[i];
        const double curr_confidence = confidences[i];
        const int curr_label = labels[i];
        VoxelKey curr_key = pointToVoxel(curr_point);

        // 2. 找到 / 创建 accumulator
        SemanticVoxel& accumulator = voxels_[curr_key];

        // 3. point_sum += point
        // 4. color_sum += color
        for (int j = 0; j < 3; j++){
            accumulator.point_sum[j] += curr_point[j];
            accumulator.color_sum[j] += curr_color[j];
        };

        // 5. count += 1
        accumulator.count += 1;

        // 6. confidence_sum += confidence
        accumulator.confidence_sum += curr_confidence;

        // 7. label_scores[label] += confidence
        accumulator.label_scores[curr_label] += curr_confidence;
    }

}


std::vector<SemanticVoxelOutput> SemanticVoxelMap::exportMap() const {
    std::vector<SemanticVoxelOutput> outputs;
    outputs.reserve(voxels_.size());

    for (const auto& pair : voxels_) {
        const VoxelKey& key = pair.first;
        const SemanticVoxel& accumulator = pair.second;

        // 1 & 2. mean point and mean color
        Vec3 mean_point{}, mean_color{};
        const Vec3 &curr_point = accumulator.point_sum;
        const Vec3 &curr_color = accumulator.color_sum;
        std::size_t voxel_count = accumulator.count;
        for (int i=0; i < 3; i++){
            mean_point[i] = curr_point[i] / static_cast<double>(voxel_count);
            mean_color[i] = curr_color[i] / static_cast<double>(voxel_count);
        }

        // 3. find winning label
        const std::unordered_map<int, double> &label_scores = accumulator.label_scores;
        int winning_label = -1;
        double winning_score = -1.0;
        for (const auto &label_score_pair : label_scores){
            const int curr_label = label_score_pair.first;
            const double curr_score = label_score_pair.second;
            if ((curr_score > winning_score) || 
                ((curr_score == winning_score) && (curr_label < winning_label))){
                winning_label = curr_label;
                winning_score = curr_score;
            }
        }

        // 4. semantic agreement
        double total_score = accumulator.confidence_sum;
        double semantic_agreement = 0.0;
        if (total_score > 0.0){
            semantic_agreement = winning_score / total_score;
        } 

        // 5. mean model confidence
        double mean_model_confidence = total_score / static_cast<double>(voxel_count);

        // 6. construct output
        SemanticVoxelOutput curr_output = SemanticVoxelOutput{
            key,
            mean_point,
            mean_color,
            winning_label,
            semantic_agreement,
            mean_model_confidence,
            voxel_count,
        };

        // 7. push_back
        outputs.push_back(curr_output);
    }

    std::sort(
        outputs.begin(),
        outputs.end(),
        [](const SemanticVoxelOutput& a,
        const SemanticVoxelOutput& b) {

            if (a.key.x != b.key.x) {
                return a.key.x < b.key.x;
            }

            if (a.key.y != b.key.y) {
                return a.key.y < b.key.y;
            }

            return a.key.z < b.key.z;
        } // [](){} same as lambda function in python
    );

    return outputs;
}


std::size_t SemanticVoxelMap::size() const {
    // TODO 5

    return voxels_.size();
}


}  // namespace rgbd_mapping