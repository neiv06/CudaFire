#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Simulation constants
constexpr int BLOCK_SIZE = 16;  // 16x16 thread blocks
constexpr float CELL_SIZE = 30.0f;  // meters per cell (matches LANDFIRE resolution)

// Cell state flags
enum CellState : uint8_t {
    UNBURNED = 0,
    BURNING = 1,
    BURNED = 2,
    NON_BURNABLE = 3  // water, rock, roads, etc.
};

// Fuel model based on Anderson 13 fuel models
// These are simplified - full implementation would use LANDFIRE FBFM40
struct FuelModel {
    float load;          // fuel load (kg/m�)
    float depth;         // fuel bed depth (m)
    float moisture_ext;  // moisture of extinction (fraction)
    float heat_content;  // heat content (kJ/kg)
    float sav_ratio;     // surface-area-to-volume ratio (1/m)
    float density;       // particle density (kg/m�)
};

// Per-cell terrain and fuel data (static, loaded from GIS data)
struct TerrainCell {
    float elevation;     // meters above sea level
    float slope;         // degrees (0-90)
    float aspect;        // degrees from north (0-360)
    uint8_t fuel_model;  // index into fuel model table
    float fuel_moisture; // dead fuel moisture content (fraction)
    float canopy_cover;  // fraction (0-1)
};

// Per-cell dynamic simulation state
struct SimCell {
    CellState state;
    float time_of_arrival;   // when fire reached this cell (seconds)
    float fire_intensity;    // fireline intensity (kW/m)
    float residence_time;    // how long cell has been burning
    float spread_rate;       // current spread rate (m/s)
};

// Wind data (can vary spatially for large domains)
struct WindField {
    float speed;      // m/s at midflame height
    float direction;  // degrees from north (direction wind is coming FROM)
};

// Main simulation grid
struct SimulationGrid {
    int width;
    int height;

    // Device pointers
    TerrainCell* d_terrain;
    SimCell* d_cells;
    SimCell* d_cells_next;  // double buffer for updates
    WindField* d_wind;
    FuelModel* d_fuel_models;

    // Host pointers (for I/O)
    TerrainCell* h_terrain;
    SimCell* h_cells;

    // Simulation parameters
    float dt;               // time step (seconds)
    float current_time;     // elapsed simulation time
    int active_cells;       // number of currently burning cells
};

// Core simulation functions
#ifdef __cplusplus
extern "C" {
#endif

    // Initialize simulation grid
    void initSimulation(SimulationGrid* grid, int width, int height);

    // Free simulation resources
    void freeSimulation(SimulationGrid* grid);

    // Copy terrain data to device
    void uploadTerrain(SimulationGrid* grid, const TerrainCell* terrain);

    // Run one simulation step
    void stepSimulation(SimulationGrid* grid, float dt);

    // Ignite cells at specified locations
    void ignitePoints(SimulationGrid* grid, const int* x_coords, const int* y_coords,
        int num_points, float time);

    // Download current state to host
    void downloadState(SimulationGrid* grid);

    // Get statistics
    int countBurningCells(SimulationGrid* grid);
    int countBurnedCells(SimulationGrid* grid);

#ifdef __cplusplus
}
#endif

// CUDA kernel declarations
__global__ void fireSpreadKernel(
    const TerrainCell* terrain,
    const SimCell* cells_current,
    SimCell* cells_next,
    const WindField* wind,
    const FuelModel* fuel_models,
    int width, int height,
    float dt, float current_time
);

__global__ void countCellsKernel(
    const SimCell* cells,
    int* burning_count,
    int* burned_count,
    int total_cells
);

#endif // SIMULATION_CUH