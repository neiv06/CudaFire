# CUDA Wildfire Spread Simulator

A GPU-accelerated wildfire spread simulator using CUDA and the Rothermel fire spread model. This project simulates how wildfires propagate across terrain based on fuel types, moisture content, slope, and wind conditions.

![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)

## Features

- **GPU-Accelerated**: Leverages NVIDIA CUDA for massively parallel fire spread calculations
- **Rothermel Model**: Industry-standard fire behavior equations used by US Forest Service
- **Anderson 13 Fuel Models**: Standard fuel classification system for fire behavior prediction
- **Realistic Physics**: Accounts for wind, slope, fuel moisture, and terrain effects
- **Synthetic Terrain Generation**: Built-in procedural terrain for testing
- **High Performance**: Simulates 6 hours of fire spread in ~3 seconds on RTX 3080

## Requirements

- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (tested on RTX 3080)
- **CUDA Toolkit**: Version 12.x
- **Compiler**: Visual Studio 2022 with MSVC
- **Build System**: CMake 3.18+ and Ninja

## Building

### Windows (Visual Studio 2022)

1. Open **x64 Native Tools Command Prompt for VS 2022** (important - not regular Command Prompt)

2. Navigate to project directory:
   ```cmd
   cd C:\path\to\cuda-wildfire-simulator
   ```

3. Build with CMake and Ninja:
   ```cmd
   mkdir build
   cd build
   cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
   ninja
   ```

4. Run the simulator:
   ```cmd
   wildfire_sim.exe
   ```

## Project Structure

```
cuda-wildfire-simulator/
├── include/
│   ├── simulation.cuh    # CUDA simulation structures and declarations
│   ├── rothermel.cuh     # Rothermel fire spread model (GPU device functions)
│   └── terrain.h         # Terrain loading and processing classes
├── src/
│   ├── main.cpp          # Entry point and simulation loop
│   ├── simulation.cu     # CUDA kernels for fire spread
│   ├── rothermel.cu      # Anderson 13 fuel model definitions
│   └── terrain.cpp       # Terrain generation and slope/aspect calculation
├── CMakeLists.txt        # CMake build configuration
└── README.md
```

## How It Works

### 1. Simulation Overview

The simulator uses a **cellular automaton** approach where the terrain is divided into a grid of cells (default 1024×1024, each 30m×30m). Each cell contains:
- Elevation, slope, and aspect (direction the slope faces)
- Fuel model (vegetation type)
- Fuel moisture content
- Fire state (unburned, burning, burned, non-burnable)

### 2. Execution Flow

```
main.cpp (entry point)
    │
    ├── printDeviceInfo()          Query GPU specs
    │
    ├── TerrainLoader
    │   ├── generateSynthetic()    Create procedural terrain
    │   └── calculateSlopeAspect() Compute slope/aspect using Horn's method
    │
    ├── initSimulation()           Allocate GPU memory (~80 MB)
    │
    ├── uploadTerrain()            Copy terrain data to GPU
    │
    ├── ignitePoints()             Set initial fire location
    │
    └── Main Loop (21,600 timesteps for 6 hours)
        │
        └── stepSimulation()
            │
            └── fireSpreadKernel<<<64×64 blocks, 16×16 threads>>>
                │
                └── 1,048,576 threads run in parallel
                    Each thread:
                    ├── Check if cell is burning
                    ├── Check 8 neighbors for fire
                    ├── Calculate spread rate (Rothermel)
                    └── Update cell state
```

### 3. Fire Spread Kernel

The core CUDA kernel (`fireSpreadKernel`) processes every cell in parallel:

1. **For burning cells**: Increment residence time, check for burnout
2. **For unburned cells**: Check all 8 neighbors for burning cells
3. **If neighbor is burning**: Calculate spread rate using Rothermel model
4. **If enough time has passed**: Ignite the cell

```cpp
// Simplified spread logic
float travel_time = distance / spread_rate;
float time_burning = current_time - neighbor.time_of_arrival;

if (time_burning >= travel_time) {
    // Fire spreads to this cell
    cell.state = BURNING;
}
```

### 4. Rothermel Fire Spread Model

The spread rate is calculated using a simplified Rothermel model:

```
R = R₀ × (1 + φ_combined)

Where:
  R₀ = base spread rate (depends on fuel type)
  φ_combined = combined wind and slope factors
```

**Key factors affecting spread rate:**
- **Fuel type**: Fine fuels (grass) spread faster than coarse fuels (timber)
- **Fuel moisture**: Wet fuels spread slower; above "moisture of extinction" won't burn
- **Slope**: Fire spreads ~6x faster up a 45° slope (φ_s = 5.275 × tan²θ)
- **Wind**: Fire spreads faster downwind (φ_w ≈ 0.6 × wind_speed)

### 5. Terrain Generation

Synthetic terrain uses multi-octave noise:
```cpp
elevation = base_elevation
    + 0.50 × range × sin(2πx) × cos(2πy)     // Large mountains
    + 0.25 × range × sin(8πx) × cos(6πy)     // Medium hills  
    + 0.15 × range × sin(20πx) × sin(20πy)   // Small ridges
    + 0.10 × range × random_noise            // Fine detail
```

Fuel models are assigned based on elevation:
- Low elevations → Grass (models 1-3)
- Mid elevations → Brush (models 5-6)
- High elevations → Timber (models 8-9)
- Very low elevations → Water (non-burnable)

## Issues Encountered During Development

### 1. CUDA Compiler Selection (LNK1181: cannot open kernel32.lib)

**Problem**: CMake was selecting MinGW/GCC instead of MSVC, causing linker errors.

**Solution**: Must use **x64 Native Tools Command Prompt for VS 2022**, not regular Command Prompt or PowerShell. This sets the correct environment variables for MSVC.

### 2. Multiple Definition Errors

**Problem**: Device functions in `rothermel.cuh` were compiled into multiple .cu files, causing linker errors.

**Solution**: Mark all `__device__` functions as `__forceinline__`:
```cpp
__device__ __forceinline__ float moistureDamping(float moisture, float moisture_ext) {
    // ...
}
```

### 3. Fire Not Spreading (Initial Implementation)

**Problem**: Fire would ignite but immediately extinguish with only 1 cell burned.

**Root Cause**: The timing logic checked `current_time - neighbor.time_of_arrival`, but both values started at 0, so `time_burning = 0` was always less than `travel_time`.

**Solution**: Use `current_time + dt` to account for the current timestep:
```cpp
float time_burning = (current_time + dt) - neighbor.time_of_arrival;
```

### 4. Fire Spreading Too Fast (1 Cell Per Timestep)

**Problem**: After fixing the timing issue, fire spread at 108 km/hr (unrealistic).

**Root Cause**: Removed timing check entirely, causing fire to spread one cell per timestep regardless of physics.

**Solution**: Kept timing check but ensured the math was correct.

### 5. Fire Extinguishing Before Spreading

**Problem**: Ignited cell burned out (~90 seconds) before fire could spread to neighbors (~375 seconds travel time).

**Root Cause**: Burnout time was too short relative to spread rate.

**Solution**: 
1. Increased burnout time: `fuel.load * 2000` (was `* 200`)
2. Increased base spread rates by 10x to match realistic fire behavior
3. Set minimum burnout time to 300 seconds (was 60)

### 6. Fuel Moisture Too High

**Problem**: Aspect-adjusted moisture could exceed moisture of extinction, preventing spread.

**Root Cause**: Initial moisture (4-12%) multiplied by aspect factor (up to 1.3x) = up to 15.6%, exceeding grass extinction threshold (12%).

**Solution**: Changed moisture calculation to maintain 3-7% range:
```cpp
cell.fuel_moisture = 0.03f + 0.04f * (1.0f + aspect_factor) * 0.5f;
```

## Sample Output

```
=== Wildfire Spread Simulator ===
CUDA-accelerated cellular automaton with Rothermel spread model

=== CUDA Device Info ===
Device: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Total Global Memory: 10239 MB
Multiprocessors: 68

Simulation Configuration:
  Grid size: 1024 x 1024 cells
  Cell resolution: 30 m
  Domain size: 30.72 x 30.72 km
  Simulation duration: 6 hours
  Timestep: 1 seconds

Running simulation...
[==================================================] 100%

=== Simulation Results ===
Wall clock time: 2.826 seconds
Simulation speed: 7643x real-time
Cells burned: 968
Area burned: 87.12 hectares (215.28 acres)
```

## Configuration

Key parameters in `main.cpp`:

```cpp
const int GRID_WIDTH = 1024;      // Grid cells (x)
const int GRID_HEIGHT = 1024;     // Grid cells (y)
const float CELL_SIZE = 30.0f;    // Meters per cell
const float SIM_DURATION = 3600.0f * 6.0f;  // 6 hours
const float DT = 1.0f;            // Timestep (seconds)

WindField wind_field = { 5.0f, 270.0f };  // 5 m/s from west
```

## Future Improvements

- [ ] Load real terrain data (GeoTIFF via GDAL)
- [ ] Load LANDFIRE fuel model data
- [ ] Export fire perimeter as GeoJSON
- [ ] Visualization (OpenGL or export to image)
- [ ] Spotting (ember transport)
- [ ] Crown fire modeling
- [ ] Variable wind fields (spatial/temporal)
- [ ] Validation against FIRMS satellite data

## References

- Rothermel, R.C. (1972). "A Mathematical Model for Predicting Fire Spread in Wildland Fuels." USDA Forest Service Research Paper INT-115.
- Anderson, H.E. (1982). "Aids to Determining Fuel Models for Estimating Fire Behavior." USDA Forest Service General Technical Report INT-122.
- LANDFIRE: https://landfire.gov/
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- US Forest Service for Rothermel model and fuel model research
- USGS for terrain data standards
