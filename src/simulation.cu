#include "simulation.cuh"
#include "rothermel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 8-connected neighborhood offsets
__constant__ int2 NEIGHBOR_OFFSETS[8] = {
    {-1, -1}, { 0, -1}, { 1, -1},
    {-1,  0},           { 1,  0},
    {-1,  1}, { 0,  1}, { 1,  1}
};

// Distance factors for diagonal vs cardinal neighbors
__constant__ float NEIGHBOR_DISTANCES[8] = {
    1.41421356f, 1.0f, 1.41421356f,
    1.0f,              1.0f,
    1.41421356f, 1.0f, 1.41421356f
};

// Direction angles to neighbors (degrees from north, clockwise)
__constant__ float NEIGHBOR_ANGLES[8] = {
    315.0f, 0.0f, 45.0f,
    270.0f,       90.0f,
    225.0f, 180.0f, 135.0f
};

// Main fire spread kernel
__global__ void fireSpreadKernel(const TerrainCell * terrain, const SimCell * cells_current, SimCell * cells_next, const WindField * wind, 
                                 const FuelModel * fuel_models, int width, int height, float dt, float current_time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Copy current state to next buffer
    SimCell current = cells_current[idx];
    SimCell next = current;

    // If cell is non-burnable or already burned, no update needed
    if (current.state == NON_BURNABLE || current.state == BURNED) {
        cells_next[idx] = next;
        return;
    }

    // Get terrain data for this cell
    TerrainCell terr = terrain[idx];

    // Get fuel model
    FuelModel fuel = fuel_models[terr.fuel_model];

    // If fuel load is zero, mark as non-burnable
    if (fuel.load <= 0.0f) {
        next.state = NON_BURNABLE;
        cells_next[idx] = next;
        return;
    }

    // Get wind at this cell (for now, uniform wind)
    WindField w = wind[0];

    // Handle burning cells - check if they've burned out
    if (current.state == BURNING) {
        next.residence_time += dt;

        // Simple burnout model: residence time based on fuel load
        // Typical residence time is 1-10 minutes depending on fuel
        float burnout_time = fuel.load * 200.0f;  // Empirical factor
        burnout_time = fmaxf(60.0f, fminf(burnout_time, 600.0f));  // Clamp 1-10 min

        if (next.residence_time >= burnout_time) {
            next.state = BURNED;
        }

        cells_next[idx] = next;
        return;
    }

    // Handle unburned cells - check if fire spreads to them
    if (current.state == UNBURNED) {
        // Check all 8 neighbors for burning cells
        float max_arrival_rate = 0.0f;

        for (int n = 0; n < 8; ++n) {
            int nx = x + NEIGHBOR_OFFSETS[n].x;
            int ny = y + NEIGHBOR_OFFSETS[n].y;

            // Bounds check
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int nidx = ny * width + nx;
            SimCell neighbor = cells_current[nidx];

            // Only burning neighbors can spread fire
            if (neighbor.state != BURNING) continue;

            // Get neighbor terrain
            TerrainCell neighbor_terr = terrain[nidx];

            // Calculate spread direction from neighbor to this cell
            float spread_dir = NEIGHBOR_ANGLES[n];
            // Invert direction (we're looking at spread INTO this cell)
            float incoming_dir = fmodf(spread_dir + 180.0f, 360.0f);

            // Calculate slope between cells
            float dz = terr.elevation - neighbor_terr.elevation;
            float dist = NEIGHBOR_DISTANCES[n] * CELL_SIZE;
            float slope_to_cell = atanf(dz / dist) * (180.0f / 3.14159265f);

            // Calculate spread rate using Rothermel
            float spread_direction;
            float spread_rate = calculateSpreadRate(
                fuel,
                terr.fuel_moisture,
                slope_to_cell,
                w.speed,
                w.direction,
                incoming_dir,
                &spread_direction
            );

            // Adjust for direction - fire spreads faster in direction of max spread
            float angle_diff = fabsf(spread_direction - incoming_dir);
            if (angle_diff > 180.0f) angle_diff = 360.0f - angle_diff;

            // Elliptical spread model - reduce rate for off-axis spread
            // Using simple cosine falloff for now
            float direction_factor = cosf(angle_diff * 3.14159265f / 180.0f);
            direction_factor = fmaxf(0.1f, direction_factor);  // Minimum 10% spread

            float adjusted_rate = spread_rate * direction_factor;

            // Time for fire to travel from neighbor to this cell
            float travel_time = dist / (adjusted_rate + 0.001f);  // Avoid div by zero

            // Check if fire can reach this cell within the timestep
            float time_since_neighbor_ignition = current_time - neighbor.time_of_arrival;

            if (time_since_neighbor_ignition >= travel_time) {
                // Fire arrives from this neighbor
                float arrival_rate = adjusted_rate;
                if (arrival_rate > max_arrival_rate) {
                    max_arrival_rate = arrival_rate;
                }
            }
        }

        // If fire arrives, ignite this cell
        if (max_arrival_rate > 0.0f) {
            next.state = BURNING;
            next.time_of_arrival = current_time;
            next.spread_rate = max_arrival_rate;
            next.residence_time = 0.0f;

            // Calculate fireline intensity
            float reaction_intensity = reactionIntensity(
                fuel,
                moistureDamping(terr.fuel_moisture, fuel.moisture_ext)
            );
            next.fire_intensity = firelineIntensity(
                max_arrival_rate,
                fuel.load * fuel.heat_content,
                reaction_intensity
            );
        }
    }
}

// Count burning/burned cells kernel (uses atomic adds)
__global__ void countCellsKernel(const SimCell* cells, int* burning_count, int* burned_count, int total_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;

    SimCell cell = cells[idx];

    if (cell.state == BURNING) {
        atomicAdd(burning_count, 1);
    }
    else if (cell.state == BURNED) {
        atomicAdd(burned_count, 1);
    }
}

// Ignition kernel
__global__ void ignitionKernel(SimCell* cells, const int* x_coords, const int* y_coords, int num_points, int width, 
                               int height, float ignition_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int x = x_coords[idx];
    int y = y_coords[idx];

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int cell_idx = y * width + x;
        cells[cell_idx].state = BURNING;
        cells[cell_idx].time_of_arrival = ignition_time;
        cells[cell_idx].residence_time = 0.0f;
        cells[cell_idx].fire_intensity = 1000.0f;  // Initial intensity
        cells[cell_idx].spread_rate = 0.0f;
    }
}

// Initialize simulation
void initSimulation(SimulationGrid* grid, int width, int height) {
    grid->width = width;
    grid->height = height;
    grid->dt = 1.0f;
    grid->current_time = 0.0f;
    grid->active_cells = 0;

    size_t terrain_size = width * height * sizeof(TerrainCell);
    size_t cells_size = width * height * sizeof(SimCell);
    size_t fuel_size = 14 * sizeof(FuelModel);  // 13 Anderson models + 1 non-burnable

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&grid->d_terrain, terrain_size));
    CUDA_CHECK(cudaMalloc(&grid->d_cells, cells_size));
    CUDA_CHECK(cudaMalloc(&grid->d_cells_next, cells_size));
    CUDA_CHECK(cudaMalloc(&grid->d_wind, sizeof(WindField)));
    CUDA_CHECK(cudaMalloc(&grid->d_fuel_models, fuel_size));

    // Allocate host memory
    grid->h_terrain = new TerrainCell[width * height];
    grid->h_cells = new SimCell[width * height];

    // Initialize cells to unburned
    for (int i = 0; i < width * height; ++i) {
        grid->h_cells[i].state = UNBURNED;
        grid->h_cells[i].time_of_arrival = -1.0f;
        grid->h_cells[i].fire_intensity = 0.0f;
        grid->h_cells[i].residence_time = 0.0f;
        grid->h_cells[i].spread_rate = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(grid->d_cells, grid->h_cells, cells_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid->d_cells_next, grid->h_cells, cells_size, cudaMemcpyHostToDevice));

    // Initialize fuel models
    FuelModel* h_fuel_models = new FuelModel[14];
    getAnderson13FuelModels(h_fuel_models);
    CUDA_CHECK(cudaMemcpy(grid->d_fuel_models, h_fuel_models, fuel_size, cudaMemcpyHostToDevice));
    delete[] h_fuel_models;

    // Initialize default wind (calm)
    WindField default_wind = { 0.0f, 0.0f };
    CUDA_CHECK(cudaMemcpy(grid->d_wind, &default_wind, sizeof(WindField), cudaMemcpyHostToDevice));

    printf("Simulation initialized: %d x %d grid (%d cells)\n", width, height, width * height);
    printf("Memory allocated: %.2f MB on device\n",
        (terrain_size + cells_size * 2 + fuel_size) / (1024.0f * 1024.0f));
}

// Free simulation resources
void freeSimulation(SimulationGrid* grid) {
    cudaFree(grid->d_terrain);
    cudaFree(grid->d_cells);
    cudaFree(grid->d_cells_next);
    cudaFree(grid->d_wind);
    cudaFree(grid->d_fuel_models);

    delete[] grid->h_terrain;
    delete[] grid->h_cells;
}

// Upload terrain data
void uploadTerrain(SimulationGrid* grid, const TerrainCell* terrain) {
    size_t size = grid->width * grid->height * sizeof(TerrainCell);
    memcpy(grid->h_terrain, terrain, size);
    CUDA_CHECK(cudaMemcpy(grid->d_terrain, terrain, size, cudaMemcpyHostToDevice));
}

// Step simulation
void stepSimulation(SimulationGrid* grid, float dt) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (grid->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (grid->height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Run spread kernel
    fireSpreadKernel << <gridDim, block >> > (
        grid->d_terrain,
        grid->d_cells,
        grid->d_cells_next,
        grid->d_wind,
        grid->d_fuel_models,
        grid->width, grid->height,
        dt, grid->current_time
        );

    CUDA_CHECK(cudaGetLastError());

    // Swap buffers
    SimCell* temp = grid->d_cells;
    grid->d_cells = grid->d_cells_next;
    grid->d_cells_next = temp;

    grid->current_time += dt;
}

// Ignite points
void ignitePoints(SimulationGrid* grid, const int* x_coords, const int* y_coords,
    int num_points, float time) {
    int* d_x, * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, num_points * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x, x_coords, num_points * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y_coords, num_points * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    ignitionKernel << <blocks, threads >> > (
        grid->d_cells, d_x, d_y, num_points,
        grid->width, grid->height, time
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_x);
    cudaFree(d_y);
}

// Download state to host
void downloadState(SimulationGrid* grid) {
    size_t size = grid->width * grid->height * sizeof(SimCell);
    CUDA_CHECK(cudaMemcpy(grid->h_cells, grid->d_cells, size, cudaMemcpyDeviceToHost));
}

// Count burning cells
int countBurningCells(SimulationGrid* grid) {
    int* d_burning, * d_burned;
    CUDA_CHECK(cudaMalloc(&d_burning, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_burned, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_burning, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_burned, 0, sizeof(int)));

    int total = grid->width * grid->height;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    countCellsKernel << <blocks, threads >> > (
        grid->d_cells, d_burning, d_burned, total
        );

    int burning;
    CUDA_CHECK(cudaMemcpy(&burning, d_burning, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_burning);
    cudaFree(d_burned);

    return burning;
}

// Count burned cells
int countBurnedCells(SimulationGrid* grid) {
    int* d_burning, * d_burned;
    CUDA_CHECK(cudaMalloc(&d_burning, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_burned, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_burning, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_burned, 0, sizeof(int)));

    int total = grid->width * grid->height;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    countCellsKernel << <blocks, threads >> > (
        grid->d_cells, d_burning, d_burned, total
        );

    int burned;
    CUDA_CHECK(cudaMemcpy(&burned, d_burned, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_burning);
    cudaFree(d_burned);

    return burned;
}
