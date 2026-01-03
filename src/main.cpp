#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

#include "simulation.cuh"
#include "terrain.h"

// Print CUDA device info
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
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Block Dimensions: " << prop.maxThreadsDim[0] << " x "
        << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << " x "
        << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "========================" << std::endl << std::endl;
}

// Simple progress bar
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
    std::cout << "CUDA-accelerated cellular automaton with Rothermel spread model" << std::endl;
    std::cout << std::endl;

    // Print device info
    printDeviceInfo();

    // Simulation parameters
    const int GRID_WIDTH = 1024;   // cells
    const int GRID_HEIGHT = 1024;  // cells
    const float CELL_SIZE = 30.0f; // meters
    const float SIM_DURATION = 3600.0f * 6.0f;  // 6 hours in seconds
    const float DT = 1.0f;  // 1 second timestep

    std::cout << "Simulation Configuration:" << std::endl;
    std::cout << "  Grid size: " << GRID_WIDTH << " x " << GRID_HEIGHT << " cells" << std::endl;
    std::cout << "  Cell resolution: " << CELL_SIZE << " m" << std::endl;
    std::cout << "  Domain size: " << (GRID_WIDTH * CELL_SIZE / 1000.0f) << " x "
        << (GRID_HEIGHT * CELL_SIZE / 1000.0f) << " km" << std::endl;
    std::cout << "  Simulation duration: " << (SIM_DURATION / 3600.0f) << " hours" << std::endl;
    std::cout << "  Timestep: " << DT << " seconds" << std::endl;
    std::cout << std::endl;

    // Load or generate terrain
    TerrainLoader terrain_loader;

    // For now, generate synthetic terrain for testing
    std::cout << "Generating synthetic terrain..." << std::endl;
    terrain_loader.generateSynthetic(GRID_WIDTH, GRID_HEIGHT, 500.0f, 300.0f);
    terrain_loader.calculateSlopeAspect();

    // Debug: Check terrain at ignition point
    const auto& terrain_data = terrain_loader.getTerrain();
    int center_idx = (GRID_HEIGHT / 2) * GRID_WIDTH + (GRID_WIDTH / 2);
    std::cout << "Debug - Center cell terrain:" << std::endl;
    std::cout << "  Fuel model: " << (int)terrain_data[center_idx].fuel_model << std::endl;
    std::cout << "  Fuel moisture: " << terrain_data[center_idx].fuel_moisture << std::endl;
    std::cout << "  Elevation: " << terrain_data[center_idx].elevation << std::endl;
    std::cout << "  Slope: " << terrain_data[center_idx].slope << std::endl;

    // Initialize simulation
    std::cout << "Initializing simulation..." << std::endl;
    SimulationGrid grid;
    initSimulation(&grid, GRID_WIDTH, GRID_HEIGHT);

    // Upload terrain to GPU
    uploadTerrain(&grid, terrain_loader.getTerrain().data());

    // Upload terrain to GPU
    uploadTerrain(&grid, terrain_loader.getTerrain().data());

    // Set wind conditions and upload to GPU
    WindField wind_field = { 10.0f, 270.0f };  // 5 m/s from the west
    cudaMemcpy(grid.d_wind, &wind_field, sizeof(WindField), cudaMemcpyHostToDevice);

    // Set ignition point(s) - center of domain
    std::vector<int> ignition_x = { GRID_WIDTH / 2 };
    std::vector<int> ignition_y = { GRID_HEIGHT / 2 };

    std::cout << "Setting ignition at (" << ignition_x[0] << ", " << ignition_y[0] << ")" << std::endl;
    ignitePoints(&grid, ignition_x.data(), ignition_y.data(),
        static_cast<int>(ignition_x.size()), 0.0f);

    // Run simulation
    std::cout << std::endl << "Running simulation..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    int total_steps = static_cast<int>(SIM_DURATION / DT);
    int report_interval = total_steps / 100;  // Report every 1%
    if (report_interval < 1) report_interval = 1;

    for (int step = 0; step < total_steps; ++step) {
        stepSimulation(&grid, DT);

        // Progress update
        if (step % report_interval == 0 || step == total_steps - 1) {
            float progress = static_cast<float>(step + 1) / total_steps;
            printProgress(progress);
        }

        // Early termination if fire is out
        if (step % 1000 == 0) {
            int burning = countBurningCells(&grid);
            if (burning == 0 && step > 0) {
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
    float burned_area_ha = burned * CELL_SIZE * CELL_SIZE / 10000.0f;  // hectares

    std::cout << "=== Simulation Results ===" << std::endl;
    std::cout << "Wall clock time: " << duration.count() / 1000.0f << " seconds" << std::endl;
    std::cout << "Simulation speed: " << (SIM_DURATION / (duration.count() / 1000.0f))
        << "x real-time" << std::endl;
    std::cout << "Cells burned: " << burned << std::endl;
    std::cout << "Cells still burning: " << burning << std::endl;
    std::cout << "Area burned: " << burned_area_ha << " hectares ("
        << (burned_area_ha * 2.47105f) << " acres)" << std::endl;
    std::cout << "===========================" << std::endl;

    // TODO: Output results
    // - Write fire perimeter as GeoJSON or Shapefile
    // - Write arrival time raster as GeoTIFF
    // - Validate against FIRMS data if available

    // Cleanup
    freeSimulation(&grid);

    std::cout << std::endl << "Simulation complete!" << std::endl;

    return 0;
}