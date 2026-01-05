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

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << "  --synthetic             Run with synthetic terrain (default)" << std::endl;
    std::cout << "  --dem <file>            Load elevation from GeoTIFF file" << std::endl;
    std::cout << "  --fuel <file>           Load fuel model from GeoTIFF file" << std::endl;
    std::cout << "  --ignition <lon,lat>    Set ignition point (longitude,latitude)" << std::endl;
    std::cout << "  --ignition-cell <x,y>   Set ignition point (grid cell coordinates)" << std::endl;
    std::cout << "  --wind <speed,dir>      Set wind (speed in m/s, direction in degrees)" << std::endl;
    std::cout << "  --duration <hours>      Simulation duration in hours (default: 6)" << std::endl;
    std::cout << "  --moisture <percent>    Override fuel moisture (default: from terrain)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " --synthetic" << std::endl;
    std::cout << "  " << program << " --dem data/elevation.tif --fuel data/fuel.tif --ignition -121.4347,39.9075" << std::endl;
    std::cout << "  " << program << " --dem data/dixie_elevation.tif --fuel data/dixie_fuel.tif --ignition -121.4347,39.9075 --wind 8,45 --duration 24" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Wildfire Spread Simulator ===" << std::endl;
    std::cout << "CUDA-accelerated cellular automaton with Rothermel spread model" << std::endl;
    std::cout << std::endl;

    // Default parameters
    bool use_synthetic = true;
    std::string dem_file = "";
    std::string fuel_file = "";
    double ignition_lon = 0.0;
    double ignition_lat = 0.0;
    int ignition_cell_x = -1;
    int ignition_cell_y = -1;
    bool ignition_from_coords = false;
    float wind_speed = 5.0f;
    float wind_dir = 270.0f;
    float sim_hours = 6.0f;
    float moisture_override = -1.0f;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--synthetic") {
            use_synthetic = true;
        }
        else if (arg == "--dem" && i + 1 < argc) {
            dem_file = argv[++i];
            use_synthetic = false;
        }
        else if (arg == "--fuel" && i + 1 < argc) {
            fuel_file = argv[++i];
        }
        else if (arg == "--ignition" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                ignition_lon = std::stod(coords.substr(0, comma));
                ignition_lat = std::stod(coords.substr(comma + 1));
                ignition_from_coords = true;
            }
        }
        else if (arg == "--ignition-cell" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                ignition_cell_x = std::stoi(coords.substr(0, comma));
                ignition_cell_y = std::stoi(coords.substr(comma + 1));
            }
        }
        else if (arg == "--wind" && i + 1 < argc) {
            std::string wind = argv[++i];
            size_t comma = wind.find(',');
            if (comma != std::string::npos) {
                wind_speed = std::stof(wind.substr(0, comma));
                wind_dir = std::stof(wind.substr(comma + 1));
            }
        }
        else if (arg == "--duration" && i + 1 < argc) {
            sim_hours = std::stof(argv[++i]);
        }
        else if (arg == "--moisture" && i + 1 < argc) {
            moisture_override = std::stof(argv[++i]) / 100.0f;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Print device info
    printDeviceInfo();

    // Load or generate terrain
    TerrainLoader terrain_loader;
    int grid_width, grid_height;
    float cell_size;

    if (use_synthetic) {
        std::cout << "Mode: Synthetic Terrain" << std::endl;
        grid_width = 1024;
        grid_height = 1024;
        cell_size = 30.0f;

        std::cout << "Generating synthetic terrain..." << std::endl;
        terrain_loader.generateSynthetic(grid_width, grid_height, 500.0f, 300.0f);
    }
    else {
        std::cout << "Mode: Real Terrain Data" << std::endl;

        if (dem_file.empty()) {
            std::cerr << "Error: --dem file required for real terrain mode" << std::endl;
            return 1;
        }

        std::cout << "Loading elevation: " << dem_file << std::endl;
        if (!terrain_loader.loadElevation(dem_file)) {
            std::cerr << "Failed to load elevation file" << std::endl;
            return 1;
        }

        if (!fuel_file.empty()) {
            std::cout << "Loading fuel model: " << fuel_file << std::endl;
            if (!terrain_loader.loadFuelModel(fuel_file)) {
                std::cerr << "Failed to load fuel model file" << std::endl;
                return 1;
            }
        }
        else {
            std::cout << "Warning: No fuel model specified, using default grass (model 1)" << std::endl;
        }

        grid_width = terrain_loader.getWidth();
        grid_height = terrain_loader.getHeight();
        cell_size = terrain_loader.getCellSize();
    }

    // Calculate slope and aspect
    terrain_loader.calculateSlopeAspect();

    // Apply moisture override if specified
    if (moisture_override >= 0.0f) {
        std::cout << "Overriding fuel moisture to " << (moisture_override * 100.0f) << "%" << std::endl;
        auto& terrain = const_cast<std::vector<TerrainCell>&>(terrain_loader.getTerrain());
        for (auto& cell : terrain) {
            if (cell.fuel_model > 0) {
                cell.fuel_moisture = moisture_override;
            }
        }
    }

    // Simulation parameters
    const float SIM_DURATION = 3600.0f * sim_hours;
    const float DT = 1.0f;

    std::cout << std::endl;
    std::cout << "Simulation Configuration:" << std::endl;
    std::cout << "  Grid size: " << grid_width << " x " << grid_height << " cells" << std::endl;
    std::cout << "  Cell resolution: " << cell_size << " m" << std::endl;
    std::cout << "  Domain size: " << (grid_width * cell_size / 1000.0f) << " x "
        << (grid_height * cell_size / 1000.0f) << " km" << std::endl;
    std::cout << "  Simulation duration: " << sim_hours << " hours" << std::endl;
    std::cout << "  Wind: " << wind_speed << " m/s from " << wind_dir << " degrees" << std::endl;
    std::cout << std::endl;

    // Initialize simulation
    std::cout << "Initializing simulation..." << std::endl;
    SimulationGrid grid;
    initSimulation(&grid, grid_width, grid_height);

    // Upload terrain to GPU
    uploadTerrain(&grid, terrain_loader.getTerrain().data());

    // Set wind conditions
    WindField wind_field = { wind_speed, wind_dir };
    cudaMemcpy(grid.d_wind, &wind_field, sizeof(WindField), cudaMemcpyHostToDevice);

    // Determine ignition point
    int ign_x, ign_y;
    if (ignition_cell_x >= 0 && ignition_cell_y >= 0) {
        // Use cell coordinates directly
        ign_x = ignition_cell_x;
        ign_y = ignition_cell_y;
    }
    else if (ignition_from_coords) {
        // Convert lat/lon to cell coordinates
        terrain_loader.getCellCoords(ignition_lon, ignition_lat, ign_x, ign_y);
        std::cout << "Ignition coordinates: (" << ignition_lon << ", " << ignition_lat << ")" << std::endl;
    }
    else {
        // Default: center of domain
        ign_x = grid_width / 2;
        ign_y = grid_height / 2;
    }

    std::cout << "Ignition cell: (" << ign_x << ", " << ign_y << ")" << std::endl;

    std::vector<int> ignition_xs = { ign_x };
    std::vector<int> ignition_ys = { ign_y };
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