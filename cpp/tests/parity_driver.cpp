#include "rgbd_mapping/semantic_voxel_map.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace rgbd_mapping;

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr
            << "Usage: parity_driver <input_csv>\n";
        return 1;
    }

    const std::string input_path = argv[1];

    double total_update_ms = 0.0;

    std::ifstream input_file(input_path);

    if (!input_file.is_open()) {
        std::cerr
            << "Failed to open: "
            << input_path
            << "\n";
        return 1;
    }

    SemanticVoxelMap map(0.02);

    std::vector<Vec3> frame_points;
    std::vector<Vec3> frame_colors;
    std::vector<int> frame_labels;
    std::vector<double> frame_confidences;

    int current_frame_id = -1;

    std::string line;

    std::getline(input_file, line); // remove the header

    while (std::getline(input_file, line)) {

        std::stringstream ss(line);
        std::string token;

        // TODO: parse 9 columns
        std::getline(ss, token, ',');
        const int frame_id = std::stoi(token);

        std::getline(ss, token, ',');
        const double x = std::stod(token);

        std::getline(ss, token, ',');
        const double y = std::stod(token);

        std::getline(ss, token, ',');
        const double z = std::stod(token);

        std::getline(ss, token, ',');
        const double r = std::stod(token);

        std::getline(ss, token, ',');
        const double g = std::stod(token);

        std::getline(ss, token, ',');
        const double b = std::stod(token);

        std::getline(ss, token, ',');
        const int label = std::stoi(token);

        std::getline(ss, token, ',');
        const double confidence = std::stod(token);

        Vec3 point{x, y, z};
        Vec3 color{r, g, b};

        if (current_frame_id == -1) {
            current_frame_id = frame_id;
        } if (current_frame_id != frame_id){
            const auto start = std::chrono::steady_clock::now();
            map.update(
                frame_points,
                frame_colors,
                frame_labels,
                frame_confidences
            );

            const auto end = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            total_update_ms += elapsed_ms;

            frame_points.clear();
            frame_colors.clear();
            frame_labels.clear();
            frame_confidences.clear();

            std::cerr
                << "Frame "
                << current_frame_id
                << ": "
                << frame_points.size()
                << " observations, "
                << map.size()
                << " voxels\n";
            
            current_frame_id = frame_id;

        } frame_points.push_back(point);
        frame_colors.push_back(color);
        frame_labels.push_back(label);
        frame_confidences.push_back(confidence);
    } if (!frame_points.empty()) {
        // process the last frame

        map.update(
            frame_points,
            frame_colors,
            frame_labels,
            frame_confidences
        );

        std::cerr
            << "Frame "
            << current_frame_id
            << ": "
            << frame_points.size()
            << " observations, "
            << map.size()
            << " voxels\n";
    }

    std::cerr
        << "Total mapping update time: "
        << total_update_ms
        << " ms\n";

    const auto outputs = map.exportMap();
    std::cout << std::setprecision(16);
    std::cout<< "px,py,pz,r,g,b,label,agreement,mean_conf,count" <<"\n";

    for (const auto& voxel : outputs) {
        std::cout
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