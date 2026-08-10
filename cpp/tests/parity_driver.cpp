#include "rgbd_mapping/semantic_voxel_map.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace rgbd_mapping;

int main(){
    SemanticVoxelMap map(0.01);

    std::vector<Vec3> points{
        { 0.001, 0.002, 0.003},
        { 0.004, 0.005, 0.006},
        { 0.011, 0.000, 0.000},
        {-0.001, 0.000, 0.000},
        { 0.003, 0.001, 0.001},
    };

    std::vector<Vec3> colors{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.5, 0.5, 0.5},
        {1.0, 1.0, 0.0},
    };

    std::vector<int> labels{1, 2, 3, 4, 1};

    std::vector<double> confidences{0.9, 0.6, 0.8, 0.7, 0.5};

    map.update(
        points,
        colors,
        labels,
        confidences
    );

    const auto outputs = map.exportMap();
    std::cout<< "key_x,key_y,key_z,px,py,pz,r,g,b,label,agreement,mean_conf,count" <<"\n";
    for (const auto& voxel : outputs) {
        std::cout
            << voxel.key.x << ","
            << voxel.key.y << ","
            << voxel.key.z << ","
            << voxel.point[0] << ","
            << voxel.point[1] << ","
            << voxel.point[2] << ","
            << voxel.rgb_color[0] << ","
            << voxel.rgb_color[1] << ","
            << voxel.rgb_color[2] << ","
            << voxel.label << ","
            << voxel.semantic_agreement << ","
            << voxel.mean_model_confidence << ","
            << voxel.observation_count
            << "\n";
    }

    return 0;
}