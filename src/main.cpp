#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#include "simulation.cuh"
#include "terrain.h"

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "=== CUDA Device Info ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "========================" << std::endl << std::endl;
}

void printProgress(float progress, int width = 50) {
    int pos = static_cast<int>(width * progress);
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(progress * 100.0f) << "%" << std::flush;
}

int main(int argc, char** argv) {
    std::cout << "=== Wildfire Spread Simulator ===" << std::endl;
    std::cout << "=== Dixie Fire Simulation ===" << std::endl;
    std::cout << std::endl;

    printDeviceInfo();

    // Data file paths
    std::string elevation_file = "data/dixie_elevation.tif";
    std::string fuel_file = "data/dixie_fuel.tif";

    // Load terrain data
    TerrainLoader terrain_loader;

    std::cout << "Loading elevation data..." << std::endl;
    if (!terrain_loader.loadElevation(elevation_file)) {
        std::cerr << "Failed to load elevation. Using synthetic terrain." << std::endl;
        terrain_loader.generateSynthetic(1024, 1024, 1000.0f, 500.0f);
    }

    std::cout << "Loading fuel model data..." << std::endl;
    if (!terrain_loader.loadFuelModel(fuel_file)) {
        std::cerr << "Failed to load fuel model." << std::endl;
        return 1;
    }

    // Calculate slope and aspect from elevation
    terrain_loader.calculateSlopeAspect();

    int grid_width = terrain_loader.getWidth();
    int grid_height = terrain_loader.getHeight();
    float cell_size = terrain_loader.getCellSize();

    // Simulation parameters
    const float SIM_DURATION = 3600.0f * 24.0f;  // 24 hours
    const float DT = 1.0f;  // 1 second timestep

    std::cout << std::endl;
    std::cout << "Simulation Configuration:" << std::endl;
    std::cout << "  Grid size: " << grid_width << " x " << grid_height << " cells" << std::endl;
    std::cout << "  Cell resolution: " << cell_size << " m" << std::endl;
    std::cout << "  Domain size: " << (grid_width * cell_size / 1000.0f) << " x "
        << (grid_height * cell_size / 1000.0f) << " km" << std::endl;
    std::cout << "  Simulation duration: " << (SIM_DURATION / 3600.0f) << " hours" << std::endl;
    std::cout << "  Total cells: " << (grid_width * grid_height) << std::endl;
    std::cout << std::endl;

    // Initialize simulation
    std::cout << "Initializing simulation..." << std::endl;
    SimulationGrid grid;
    initSimulation(&grid, grid_width, grid_height);

    // Upload terrain to GPU
    uploadTerrain(&grid, terrain_loader.getTerrain().data());

    // Set wind conditions - Dixie Fire had variable winds, using average
    // Northeast wind pushing fire southwest initially
    WindField wind_field = { 8.0f, 45.0f };  // 8 m/s from northeast
    cudaMemcpy(grid.d_wind, &wind_field, sizeof(WindField), cudaMemcpyHostToDevice);
    std::cout << "Wind: " << wind_field.speed << " m/s from " << wind_field.direction << " degrees" << std::endl;

    // Dixie Fire ignition point - near Cresta Dam
    // Coordinates: 39.9075°N, 121.4347°W
    // Convert to grid coordinates
    double ignition_lat = 39.9075;
    double ignition_lon = -121.4347;
    int ignition_x, ignition_y;
    terrain_loader.getCellCoords(ignition_lon, ignition_lat, ignition_x, ignition_y);

    std::cout << "Ignition point: (" << ignition_lon << ", " << ignition_lat << ")" << std::endl;
    std::cout << "  Grid coords: (" << ignition_x << ", " << ignition_y << ")" << std::endl;

    std::vector<int> ignition_xs = { ignition_x };
    std::vector<int> ignition_ys = { ignition_y };
    ignitePoints(&grid, ignition_xs.data(), ignition_ys.data(),
        static_cast<int>(ignition_xs.size()), 0.0f);

    // Run simulation
    std::cout << std::endl << "Running simulation..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    int total_steps = static_cast<int>(SIM_DURATION / DT);
    int report_interval = total_steps / 100;
    if (report_interval < 1) report_interval = 1;

    for (int step = 0; step < total_steps; ++step) {
        stepSimulation(&grid, DT);

        if (step % report_interval == 0 || step == total_steps - 1) {
            float progress = static_cast<float>(step + 1) / total_steps;
            printProgress(progress);
        }

        // Check fire status periodically
        if (step % 10000 == 0 && step > 0) {
            int burning = countBurningCells(&grid);
            if (burning == 0) {
                std::cout << std::endl << "Fire extinguished at t="
                    << (step * DT / 3600.0f) << " hours" << std::endl;
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << std::endl << std::endl;

    // Download final state
    downloadState(&grid);

    // Print statistics
    int burned = countBurnedCells(&grid);
    int burning = countBurningCells(&grid);
    float burned_area_ha = burned * cell_size * cell_size / 10000.0f;

    std::cout << "=== Simulation Results ===" << std::endl;
    std::cout << "Wall clock time: " << duration.count() / 1000.0f << " seconds" << std::endl;
    std::cout << "Simulation speed: " << (SIM_DURATION / (duration.count() / 1000.0f))
        << "x real-time" << std::endl;
    std::cout << "Cells burned: " << burned << std::endl;
    std::cout << "Cells still burning: " << burning << std::endl;
    std::cout << "Area burned: " << burned_area_ha << " hectares ("
        << (burned_area_ha * 2.47105f) << " acres)" << std::endl;
    std::cout << "===========================" << std::endl;

    // Cleanup
    freeSimulation(&grid);

    std::cout << std::endl << "Simulation complete!" << std::endl;

    return 0;
}