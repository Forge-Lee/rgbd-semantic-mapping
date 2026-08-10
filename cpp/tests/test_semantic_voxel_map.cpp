#include "rgbd_mapping/semantic_voxel_map.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace rgbd_mapping;

bool nearlyEqual(
    double a,
    double b,
    double eps = 1e-9
) {
    return std::abs(a - b) < eps;
}

void testBasicFusionAndOrdering() {
    SemanticVoxelMap map(0.01);

    std::vector<Vec3> points{
        { 0.011, 0.000, 0.0},  // key ( 1, 0, 0)
        { 0.004, 0.002, 0.0},  // key ( 0, 0, 0)
        {-0.001, 0.000, 0.0},  // key (-1, 0, 0)
        { 0.001, 0.000, 0.0}   // key ( 0, 0, 0)
    };

    std::vector<Vec3> colors{
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.2, 0.2, 0.2},
        {1.0, 0.0, 0.0}
    };

    std::vector<int> labels{
        2, 1, 3, 1
    };

    std::vector<double> confidences{
        0.6, 0.7, 0.5, 0.9
    };

    map.update(
        points,
        colors,
        labels,
        confidences
    );

    const auto outputs = map.exportMap();

    assert(outputs.size() == 3);

    assert(outputs[0].key.x == -1);
    assert(outputs[0].key.y == 0);
    assert(outputs[0].key.z == 0);

    assert(outputs[1].key.x == 0);
    assert(outputs[1].key.y == 0);
    assert(outputs[1].key.z == 0);

    assert(outputs[2].key.x == 1);
    assert(outputs[2].key.y == 0);
    assert(outputs[2].key.z == 0);

    const SemanticVoxelOutput& voxel = outputs[1];

    assert(nearlyEqual(
        voxel.point[0],
        0.0025
    ));

    assert(nearlyEqual(
        voxel.point[1],
        0.001
    ));

    assert(nearlyEqual(
        voxel.point[2],
        0.0
    ));

    assert(voxel.label == 1);

    assert(nearlyEqual(
        voxel.semantic_agreement,
        1.0
    ));

    assert(nearlyEqual(
        voxel.mean_model_confidence,
        0.8
    ));

    assert(
        voxel.observation_count == 2
    );
}

void testConflictingLabels() {
    SemanticVoxelMap map(0.01);
    std::vector<Vec3> points{{0.011, 0.000, 0.0}};
    std::vector<Vec3> colors{{0.0, 0.0, 1.0}};

    std::vector<int> label_1{3};
    std::vector<int> label_2{7};
    std::vector<double> confidence_1{0.9};
    std::vector<double> confidence_2{0.6};

    map.update(
        points,
        colors,
        label_1,
        confidence_1
    );

    map.update(
        points,
        colors,
        label_2,
        confidence_2
    );

    const auto outputs = map.exportMap();

    const SemanticVoxelOutput& output = outputs[0];

    assert(output.label == 3);

    assert(nearlyEqual(
        output.semantic_agreement,
        0.6
    ));

    assert(nearlyEqual(
        output.mean_model_confidence,
        0.75
    ));

    assert(output.observation_count == 2);
}

void testTieBreaking(){
    SemanticVoxelMap map(0.01);
    std::vector<Vec3> points{{0.011, 0.000, 0.0}};
    std::vector<Vec3> colors{{0.0, 0.0, 1.0}};

    std::vector<int> label_1{8};
    std::vector<int> label_2{5};
    std::vector<double> confidence_1{0.6};
    std::vector<double> confidence_2{0.6};

    map.update(
        points,
        colors,
        label_1,
        confidence_1
    );

    map.update(
        points,
        colors,
        label_2,
        confidence_2
    );

    const auto outputs = map.exportMap();

    const SemanticVoxelOutput& output = outputs[0];

    assert(output.label == 5);

    assert(nearlyEqual(
        output.semantic_agreement,
        0.5
    ));
}

void testEmptyMap() {
    SemanticVoxelMap map(0.01);

    const auto outputs =
        map.exportMap();

    assert(outputs.empty());
    assert(map.size() == 0);
}

int main() {
    testBasicFusionAndOrdering();
    testConflictingLabels();
    testTieBreaking();
    testEmptyMap();

    std::cout << "Test passed!" << std::endl;

    return 0;
}