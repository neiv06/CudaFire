![simulator screenshot.png](https://github.com/neiv06/cuda-wildfire-simulator/blob/master/simulator%20screenshot.png | width=200)
# CUDA Wildfire Spread Simulator

A GPU-accelerated wildfire spread simulator using CUDA and the Rothermel fire spread model. This project simulates how wildfires propagate across terrain based on fuel types, moisture content, slope, and wind conditions.

![CUDA](https://img.shields.io/badge/CUDA-12.9-green.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)

## Features

### Core Simulation
- **GPU-Accelerated**: Leverages NVIDIA CUDA for massively parallel fire spread calculations using 16×16 thread blocks
- **Rothermel Fire Spread Model**: Industry-standard fire behavior equations used by US Forest Service
  - Reaction intensity calculations
  - Moisture damping coefficients
  - Wind and slope factor calculations
  - Vector-based spread direction computation
- **Anderson 13 Fuel Models**: Complete implementation of standard fuel classification system
  - Models 1-13: Grass, brush, timber, slash, and chaparral fuels
  - Fuel load, depth, moisture of extinction, heat content, SAV ratio
- **Cellular Automaton**: 8-connected neighborhood fire spread with realistic timing
- **High Performance**: Simulates 6 hours of fire spread in ~3 seconds on RTX 3080 (7643x real-time)

### Terrain Support
- **Real Terrain Loading**: Load elevation data from GeoTIFF files via GDAL
  - Automatic georeferencing and coordinate transformation
  - Support for NoData values and nodata handling
- **LANDFIRE Fuel Models**: Load fuel model grids from LANDFIRE FBFM13 GeoTIFF files
  - Automatic mapping of LANDFIRE codes to Anderson fuel models
  - Non-burnable area detection (water, snow, urban, barren)
- **Synthetic Terrain Generation**: Built-in procedural terrain for testing
  - Multi-octave noise generation (large mountains, medium hills, small ridges)
  - Elevation-based fuel model assignment
  - Configurable base elevation and elevation range
- **Slope and Aspect Calculation**: Horn's method using 3×3 window
  - Accurate slope computation in degrees
  - Aspect calculation (direction slope faces, 0-360°)
  - Aspect-adjusted fuel moisture (north-facing slopes retain more moisture)

### Fire Physics
- **Wind Effects**: Uniform wind field support
  - Wind speed and direction (meteorological convention)
  - Wind factor (φ_w) calculation from Rothermel equations
  - Vector combination with slope effects
- **Slope Effects**: Realistic uphill/downhill spread
  - Slope factor (φ_s) = 5.275 × tan²(slope) for uphill
  - Reduced spread rate for downhill
  - Directional spread based on aspect
- **Fuel Moisture**: Dynamic moisture calculations
  - Aspect-adjusted moisture (3-7% range for fire weather)
  - Moisture of extinction per fuel model
  - Moisture damping coefficient (η_m)
- **Fire Intensity**: Byram's fireline intensity calculation
  - Reaction intensity (IR) from fuel properties
  - Flame depth estimation
  - Intensity tracking per cell
- **Burnout Model**: Realistic fuel consumption
  - Residence time based on fuel load
  - Minimum/maximum burnout times (5-60 minutes)
  - Automatic state transitions (burning → burned)

### Visualization
- **OpenGL Real-Time Viewer**: Interactive 3D visualization
  - Terrain mesh rendering with elevation-based coloring
  - Real-time fire state overlay (burning = yellow/orange/red, burned = dark gray)
  - Dynamic color updates based on residence time
  - Depth testing and anti-aliasing
- **Interactive Camera Controls**:
  - WASD movement
  - Q/E for vertical movement
  - Mouse look (Tab to toggle)
  - Scroll wheel for speed adjustment
  - Shift for fast movement
- **Simulation Controls**:
  - Space: Pause/Resume simulation
  - +/-: Adjust simulation speed (2x, 4x, 8x, etc.)
  - Esc: Exit viewer

### Command-Line Interface
- **Flexible Input Options**:
  - `--synthetic`: Use procedural terrain (default)
  - `--dem <file>`: Load elevation from GeoTIFF
  - `--fuel <file>`: Load fuel models from GeoTIFF
  - `--ignition <lon,lat>`: Set ignition point by geographic coordinates
  - `--ignition-cell <x,y>`: Set ignition point by grid coordinates
  - `--wind <speed,dir>`: Set wind conditions (m/s, degrees from north)
  - `--duration <hours>`: Simulation duration (default: 6 hours)
  - `--moisture <percent>`: Override fuel moisture content
  - `--visualize` or `-v`: Enable real-time OpenGL visualization
  - `--help`: Display usage information

### Data Processing
- **Coordinate Transformations**: Geographic ↔ Grid cell conversions
  - Latitude/longitude to cell coordinates
  - Cell coordinates to latitude/longitude
  - Automatic georeferencing from GeoTIFF metadata
- **Statistics Tracking**:
  - Burning cell count
  - Burned cell count
  - Area burned (hectares and acres)
  - Simulation speed (real-time multiplier)
  - Fire extinction detection
- **Progress Reporting**: Real-time progress bar during simulation

### Validation Support (Framework)
- **FIRMS Data Loading**: Framework for loading NASA FIRMS satellite fire detections
  - CSV parsing for fire detection data
  - Time range filtering
  - Grid cell conversion for comparison
  - Jaccard index calculation for validation metrics

### Technical Features
- **Double Buffering**: GPU memory double buffering for state updates
- **CUDA Error Checking**: Comprehensive error handling and reporting
- **Memory Management**: Efficient GPU memory allocation (~80 MB for 1024×1024 grid)
- **Separable Compilation**: CUDA separable compilation enabled
- **Fast Math**: CUDA fast math optimizations enabled
- **Line Info**: CUDA line info for debugging support

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (tested on RTX 3080, Compute Capability 8.6)
- **Memory**: ~80 MB GPU memory for 1024×1024 grid (scales with grid size)
- **Display**: For visualization mode, requires OpenGL 3.3+ capable graphics

### Software
- **CUDA Toolkit**: Version 12.x (tested with 12.9)
- **Compiler**: Visual Studio 2022 with MSVC (C++17 support)
- **Build System**: CMake 3.18+ and Ninja
- **Libraries**:
  - **GDAL**: For GeoTIFF terrain loading (set via CMake command line)
  - **OpenGL**: For visualization (via system OpenGL)
  - **GLFW**: Version 3.4+ (set path in CMakeLists.txt)
  - **GLM**: Header-only math library (set path in CMakeLists.txt)
  - **GLAD**: OpenGL loader (included in project)

## Building

### Windows (Visual Studio 2022)

1. **Install Dependencies:**
   - Install CUDA Toolkit 12.x
   - Install GDAL (or build from source)
   - Download GLFW 3.4+ and extract to `C:/libs/glfw-3.4.bin.WIN64`
   - Download GLM and extract to `C:/libs/glm`
   - Update paths in `CMakeLists.txt` if different

2. **Open Build Environment:**
   - Open **x64 Native Tools Command Prompt for VS 2022** (important - not regular Command Prompt)
   - This sets the correct environment variables for MSVC

3. **Configure GDAL (if needed):**
   ```cmd
   set GDAL_DIR=C:\path\to\gdal
   set GDAL_INCLUDE_DIR=%GDAL_DIR%\include
   set GDAL_LIBRARY=%GDAL_DIR%\lib\gdal.lib
   ```

4. **Build:**
   ```cmd
   cd C:\path\to\cuda-wildfire-simulator
   mkdir build
   cd build
   cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
   ninja
   ```

5. **Run the simulator:**
   ```cmd
   wildfire_sim.exe --synthetic
   wildfire_sim.exe --help
   ```

### Build Options

- **Release mode** (default): Optimized performance, no debug symbols
- **Debug mode**: `-DCMAKE_BUILD_TYPE=Debug` for debugging with line info
- **CUDA Architecture**: Set in `CMakeLists.txt` (default: 86 for RTX 3080)
  - For other GPUs, change `CMAKE_CUDA_ARCHITECTURES` to match your GPU

## Project Structure

```
cuda-wildfire-simulator/
├── include/
│   ├── simulation.cuh    # CUDA simulation structures and declarations
│   ├── rothermel.cuh     # Rothermel fire spread model (GPU device functions)
│   ├── terrain.h         # Terrain loading and processing classes
│   ├── gl_viewer.h       # OpenGL visualization class
│   └── include/
│       ├── glad/         # OpenGL loader (GLAD)
│       └── KHR/          # Khronos platform headers
├── src/
│   ├── main.cpp          # Entry point, CLI parsing, simulation loop
│   ├── simulation.cu     # CUDA kernels for fire spread
│   ├── rothermel.cu      # Anderson 13 fuel model definitions
│   ├── terrain.cpp       # Terrain generation, GeoTIFF loading, slope/aspect
│   ├── gl_viewer.cpp     # OpenGL rendering, camera controls, shaders
│   └── glad.c            # OpenGL function loader implementation
├── CMakeLists.txt        # CMake build configuration
├── CMakeSettings.json    # Visual Studio CMake settings
├── build/                # Build directory (generated)
├── data/                 # Sample terrain data (GeoTIFF files)
└── README.md
```

## How It Works

### 1. Simulation Overview

The simulator uses a **cellular automaton** approach where the terrain is divided into a grid of cells (default 1024×1024, each 30m×30m). Each cell contains:

**Static Terrain Data (TerrainCell):**
- Elevation (meters above sea level)
- Slope (degrees, 0-90)
- Aspect (degrees from north, 0-360, direction slope faces downhill)
- Fuel model (0-13, Anderson fuel model index)
- Fuel moisture content (fraction, 0-1)
- Canopy cover (fraction, 0-1)

**Dynamic Fire State (SimCell):**
- Cell state (UNBURNED, BURNING, BURNED, NON_BURNABLE)
- Time of arrival (when fire reached this cell)
- Fire intensity (kW/m, fireline intensity)
- Residence time (how long cell has been burning)
- Spread rate (m/s, current spread rate)

### 2. Execution Flow

```
main.cpp (entry point)
    │
    ├── Parse command-line arguments
    │   ├── Terrain mode (synthetic vs. real data)
    │   ├── Ignition point (lat/lon or cell coords)
    │   ├── Wind conditions
    │   └── Simulation parameters
    │
    ├── printDeviceInfo()          Query GPU specs (name, compute capability, memory)
    │
    ├── TerrainLoader
    │   ├── generateSynthetic()    Create procedural terrain (if --synthetic)
    │   ├── loadElevation()        Load GeoTIFF DEM (if --dem)
    │   ├── loadFuelModel()        Load LANDFIRE fuel models (if --fuel)
    │   └── calculateSlopeAspect() Compute slope/aspect using Horn's method
    │
    ├── initSimulation()           Allocate GPU memory (~80 MB for 1024×1024)
    │   ├── Allocate terrain buffer
    │   ├── Allocate cell state buffers (double buffering)
    │   ├── Allocate wind field
    │   └── Initialize Anderson 13 fuel models
    │
    ├── uploadTerrain()            Copy terrain data to GPU
    │
    ├── Set wind field              Copy wind conditions to GPU
    │
    ├── ignitePoints()             Set initial fire location(s)
    │
    └── Main Loop (21,600 timesteps for 6 hours @ 1s timestep)
        │
        ├── [If visualization mode]
        │   ├── GLViewer::handleInput()    Process keyboard/mouse
        │   ├── stepSimulation()          Run N steps based on speed
        │   ├── downloadState()            Copy fire state to host
        │   ├── updateFireState()          Update OpenGL color buffer
        │   └── render()                   Draw frame
        │
        └── [If batch mode]
            ├── stepSimulation()          Run one timestep
            ├── Progress reporting        Every 1% of simulation
            └── Fire extinction check     Every 10,000 steps
                │
                └── fireSpreadKernel<<<64×64 blocks, 16×16 threads>>>
                    │
                    └── 1,048,576 threads run in parallel
                        Each thread processes one cell:
                        ├── If BURNING: Check for burnout
                        ├── If UNBURNED: Check 8 neighbors
                        ├── Calculate spread rate (Rothermel)
                        ├── Check timing (travel time vs. burn time)
                        └── Update cell state
```

### 3. Fire Spread Kernel

The core CUDA kernel (`fireSpreadKernel`) processes every cell in parallel:

**For BURNING cells:**
1. Increment residence time by `dt`
2. Calculate burnout time: `fuel.load * 2000` seconds (clamped to 5-60 minutes)
3. If residence time ≥ burnout time: transition to BURNED state

**For UNBURNED cells:**
1. Check all 8 neighbors (N, NE, E, SE, S, SW, W, NW)
2. For each BURNING neighbor:
   - Calculate slope between cells: `atan(dz / distance)`
   - Calculate spread direction from neighbor to this cell
   - Compute spread rate using Rothermel model:
     - Base rate (R₀) from fuel model
     - Moisture damping (η_m)
     - Slope factor (φ_s)
     - Wind factor (φ_w)
     - Vector combination for direction
   - Apply elliptical spread model (reduce rate for off-axis spread)
   - Calculate travel time: `distance / adjusted_rate`
   - Check if enough time has passed: `(current_time + dt) - neighbor.time_of_arrival ≥ travel_time`
3. If any neighbor can ignite this cell: transition to BURNING state
   - Record time of arrival
   - Calculate fireline intensity
   - Store spread rate

**Key Implementation Details:**
- Uses constant memory for neighbor offsets, distances, and angles
- Double buffering prevents race conditions
- Elliptical spread model: `direction_factor = cos(angle_diff)` with 10% minimum
- Distance factors: 1.0 for cardinal, 1.414 for diagonal neighbors

### 4. Rothermel Fire Spread Model

The spread rate is calculated using the Rothermel 1972 fire spread model:

```
R = R₀ × (1 + φ_combined)

Where:
  R₀ = base spread rate × moisture_damping (depends on fuel type)
  φ_combined = √(φ_slope² + φ_wind² + 2×φ_slope×φ_wind×cos(θ_diff))
```

**Base Spread Rate (R₀):**
- Fine fuels (grass, SAV > 2500): 0.05 m/s (~180 m/hr)
- Medium fuels (brush, SAV > 1800): 0.03 m/s (~108 m/hr)
- Coarse fuels (timber, SAV ≤ 1800): 0.015 m/s (~54 m/hr)
- Applied moisture damping: `R₀ × η_m`

**Moisture Damping (η_m):**
- Polynomial function: `1 - 2.59×rm + 5.11×rm² - 3.52×rm³`
- Where `rm = moisture / moisture_of_extinction`
- Returns 0 if moisture ≥ extinction threshold

**Slope Factor (φ_s):**
- Uphill: `φ_s = 5.275 × tan²(slope)` (capped at 10x)
- Downhill: `φ_s = -0.5 × |slope| / 45°` (reduces rate by up to 50%)
- Direction: Points upslope (opposite of aspect)

**Wind Factor (φ_w):**
- Simplified: `φ_w = 0.6 × wind_speed` (capped at 15x)
- Full Rothermel uses: `C × wind^B × (β/0.0012)^-E`
- Direction: Points downwind (opposite of wind direction)

**Vector Combination:**
- Slope and wind vectors are combined
- Maximum spread direction: `atan2(total_y, total_x)`
- Elliptical spread: rate reduced by `cos(angle_diff)` for off-axis spread

**Key factors affecting spread rate:**
- **Fuel type**: Fine fuels (grass) spread faster than coarse fuels (timber)
- **Fuel moisture**: Wet fuels spread slower; above "moisture of extinction" won't burn
- **Slope**: Fire spreads ~6x faster up a 45° slope
- **Wind**: Fire spreads faster downwind (approximately linear with wind speed)
- **Direction**: Fire spreads fastest in direction of combined slope+wind vector

### 5. Terrain Generation

**Synthetic Terrain:**
Uses multi-octave noise for realistic terrain:
```cpp
elevation = base_elevation
    + 0.50 × range × sin(2πx) × cos(2πy)     // Large mountains
    + 0.25 × range × sin(8πx) × cos(6πy)     // Medium hills  
    + 0.15 × range × sin(20πx) × sin(20πy)   // Small ridges
    + 0.10 × range × random_noise            // Fine detail
```

**Fuel Model Assignment:**
- Very low elevations (< -0.3×range): Water (non-burnable, model 0)
- Low elevations (-0.3 to 0): Grass (models 1-3, random)
- Mid elevations (0 to 0.3): Brush (models 5-6, random)
- High elevations (> 0.3): Timber (models 8-9, random)

**Fuel Moisture:**
- Random initial moisture: 4-12%
- Adjusted by aspect: North-facing slopes retain more moisture
- Final range: 3-7% (typical fire weather conditions)

**Canopy Cover:**
- Timber fuels (8-10): 40% ± 15%
- Brush fuels (4-7): 20% ± 10%
- Grass fuels (1-3): < 10%

### 6. Real Terrain Loading

**GeoTIFF Elevation:**
- Uses GDAL library for reading GeoTIFF files
- Supports any projection (automatically extracts geotransform)
- Handles NoData values (marked as non-burnable)
- Calculates cell size from georeferencing (meters)

**LANDFIRE Fuel Models:**
- Reads FBFM13 (Fire Behavior Fuel Model 13) GeoTIFF files
- Maps LANDFIRE codes to Anderson fuel models:
  - Codes 1-13: Direct mapping to Anderson models
  - Codes 91-99: Non-burnable (water, snow, urban, barren)
  - NoData/-9999: Non-burnable
- Validates dimensions match elevation grid

**Coordinate Transformations:**
- Geographic coordinates (lat/lon) ↔ Grid cell coordinates
- Uses geotransform matrix from GeoTIFF
- Handles edge cases (clamps to grid bounds)

## Problems & Solutions Encountered During Development

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

## Visualization

The OpenGL viewer provides real-time 3D visualization of the fire simulation:

### Visual Features
- **Terrain Rendering**: 3D mesh with elevation-based coloring
  - Green → Olive → Brown gradient based on elevation
  - Water/non-burnable areas shown in dark blue
- **Fire State Visualization**:
  - **Burning**: Yellow → Orange → Red gradient based on residence time
  - **Burned**: Dark gray/black
  - **Unburned**: Original terrain color
- **Dynamic Updates**: Fire colors update in real-time as simulation progresses
- **Smooth Rendering**: Depth testing, anti-aliasing, and perspective projection

### Camera System
- **Free-fly camera** with first-person controls
- **Initial position**: Above terrain center, angled view
- **Movement**: WASD for horizontal, Q/E for vertical
- **Mouse look**: Tab to toggle, smooth rotation
- **Speed control**: Scroll wheel or Shift for fast movement

### Performance
- Renders 1024×1024 terrain mesh (2M triangles) at 60+ FPS
- Efficient GPU buffer updates (only modified colors)
- Real-time simulation integration (runs simulation steps between frames)

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

### Command-Line Parameters

All configuration is done via command-line arguments:

```bash
# Synthetic terrain with default settings
wildfire_sim.exe --synthetic

# Real terrain with custom ignition and wind
wildfire_sim.exe --dem data/elevation.tif --fuel data/fuel.tif \
    --ignition -121.4347,39.9075 --wind 8,45 --duration 24

# Visualization mode
wildfire_sim.exe --synthetic --visualize

# Override fuel moisture
wildfire_sim.exe --dem data/elevation.tif --moisture 5.0
```

### Programmatic Configuration

Key parameters in `simulation.cuh` and `main.cpp`:

```cpp
// Grid configuration (can be overridden by terrain loading)
const int GRID_WIDTH = 1024;      // Grid cells (x)
const int GRID_HEIGHT = 1024;     // Grid cells (y)
const float CELL_SIZE = 30.0f;    // Meters per cell (LANDFIRE resolution)
const float SIM_DURATION = 3600.0f * 6.0f;  // 6 hours (default)
const float DT = 1.0f;            // Timestep (seconds)

// CUDA kernel configuration
constexpr int BLOCK_SIZE = 16;    // 16×16 thread blocks

// Wind field (can be set via --wind)
WindField wind_field = { 5.0f, 270.0f };  // 5 m/s from west (270°)
```

### Fuel Model Parameters

Anderson 13 fuel models are defined in `rothermel.cu` with:
- Fuel load (kg/m²)
- Fuel bed depth (m)
- Moisture of extinction (fraction)
- Heat content (kJ/kg)
- Surface-area-to-volume ratio (1/m)
- Particle density (kg/m³)

## Future Improvements

- [x] Load real terrain data (GeoTIFF via GDAL) ✅
- [x] Load LANDFIRE fuel model data ✅
- [x] OpenGL visualization ✅
- [ ] Export fire perimeter as GeoJSON/KML
- [ ] Export simulation results as GeoTIFF
- [ ] Spotting (ember transport and long-range ignition)
- [ ] Crown fire modeling (transition from surface to crown fire)
- [ ] Variable wind fields (spatial/temporal from weather models)
- [ ] Load wind from GRIB files (HRRR, GFS)
- [ ] Fuel moisture from NFMD (National Fuel Moisture Database)
- [ ] Canopy cover effects on wind and spread
- [ ] Validation against FIRMS satellite data (framework exists)
- [ ] Multi-GPU support for very large domains
- [ ] Adaptive timestep based on spread rate
- [ ] Fire suppression modeling
- [ ] Historical fire comparison mode

## Anderson 13 Fuel Models

The simulator implements all 13 Anderson fuel models used by the US Forest Service:

| Model | Type | Description | Typical Use |
|-------|------|-------------|-------------|
| 1 | Grass | Short grass (1 ft) | Grasslands, prairies |
| 2 | Grass | Timber grass/understory | Open timber |
| 3 | Grass | Tall grass (2.5 ft) | Tall grass prairies |
| 4 | Brush | Chaparral | Mediterranean shrublands |
| 5 | Brush | Brush (2 ft) | Dense brush |
| 6 | Brush | Dormant brush | Winter brush |
| 7 | Brush | Southern rough | Southern pine rough |
| 8 | Timber | Compact timber litter | Dense conifer litter |
| 9 | Timber | Hardwood litter | Deciduous forest litter |
| 10 | Timber | Timber understory | Closed timber |
| 11 | Slash | Light slash | Light logging slash |
| 12 | Slash | Medium slash | Medium logging slash |
| 13 | Slash | Heavy slash | Heavy logging slash |

Each fuel model has specific properties:
- **Fuel load**: 0.166-1.569 kg/m²
- **Fuel depth**: 0.061-1.829 m
- **Moisture of extinction**: 12-40%
- **Surface-area-to-volume ratio**: 1500-3500 1/m

## Usage Examples

### Basic Synthetic Simulation
```bash
# Run 6-hour simulation on synthetic terrain
wildfire_sim.exe --synthetic

# With custom duration
wildfire_sim.exe --synthetic --duration 12
```

### Real Terrain Simulation
```bash
# Load real terrain data (Dixie Fire area example)
wildfire_sim.exe --dem data/dixie_elevation.tif \
    --fuel data/dixie_fuel.tif \
    --ignition -121.4347,39.9075 \
    --wind 8,45 \
    --duration 24

# Use cell coordinates instead of lat/lon
wildfire_sim.exe --dem data/elevation.tif \
    --ignition-cell 512,512 \
    --wind 5,270
```

### Visualization Mode
```bash
# Real-time 3D visualization
wildfire_sim.exe --synthetic --visualize

# With real terrain
wildfire_sim.exe --dem data/elevation.tif --fuel data/fuel.tif --visualize
```

### Custom Moisture Conditions
```bash
# Override fuel moisture (dry conditions)
wildfire_sim.exe --dem data/elevation.tif --moisture 3.0

# Wet conditions
wildfire_sim.exe --dem data/elevation.tif --moisture 15.0
```

## References

- Rothermel, R.C. (1972). "A Mathematical Model for Predicting Fire Spread in Wildland Fuels." USDA Forest Service Research Paper INT-115.
- Anderson, H.E. (1982). "Aids to Determining Fuel Models for Estimating Fire Behavior." USDA Forest Service General Technical Report INT-122.
- LANDFIRE: https://landfire.gov/ (Fuel model and terrain data)
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/ (Satellite fire detection data)
- GDAL: https://gdal.org/ (Geospatial data abstraction library)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- US Forest Service for Rothermel model and fuel model research
- USGS for terrain data standards
