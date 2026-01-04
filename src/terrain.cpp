#include "terrain.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <gdal_priv.h>
#include <cpl_conv.h>

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// Generate synthetic terrain for testing
void TerrainLoader::generateSynthetic(int width, int height, float base_elevation, float elevation_range) {
    width_ = width;
    height_ = height;
    cell_size_ = 30.0f;

    // Resize vectors
    terrain_.resize(width * height);
    elevation_raw_.resize(width * height);

    // Random number generator for noise
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
    std::uniform_int_distribution<int> fuel_dist(1, 10);  // Fuel models 1-10
    std::uniform_real_distribution<float> moisture_dist(0.04f, 0.12f);

    std::cout << "Generating " << width << "x" << height << " synthetic terrain..." << std::endl;

    // Generate elevation using multiple octaves of Perlin-like noise
    // Simplified: use combination of sinusoids with different frequencies
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            // Multi-frequency elevation (poor man's Perlin)
            float fx = static_cast<float>(x) / width;
            float fy = static_cast<float>(y) / height;

            float elev = base_elevation;

            // Large-scale terrain features
            elev += elevation_range * 0.5f * sinf(fx * 2.0f * PI) * cosf(fy * 2.0f * PI);

            // Medium-scale hills
            elev += elevation_range * 0.25f * sinf(fx * 8.0f * PI + 1.5f) * cosf(fy * 6.0f * PI);

            // Small-scale ridges
            elev += elevation_range * 0.15f * sinf(fx * 20.0f * PI) * sinf(fy * 20.0f * PI);

            // Random noise
            elev += elevation_range * 0.1f * noise(rng);

            elevation_raw_[idx] = elev;

            // Initialize terrain cell
            TerrainCell& cell = terrain_[idx];
            cell.elevation = elev;
            cell.slope = 0.0f;  // Will be calculated
            cell.aspect = 0.0f; // Will be calculated

            // Assign fuel model based on elevation and random variation
            // Lower elevations: grass, higher: brush/timber
            float normalized_elev = (elev - base_elevation) / elevation_range;

            if (normalized_elev < -0.3f) {
                cell.fuel_model = 1;  // Short grass
            }
            else if (normalized_elev < 0.0f) {
                cell.fuel_model = fuel_dist(rng) <= 5 ? 2 : 3;  // Timber grass or tall grass
            }
            else if (normalized_elev < 0.3f) {
                cell.fuel_model = fuel_dist(rng) <= 7 ? 5 : 6;  // Brush
            }
            else {
                cell.fuel_model = fuel_dist(rng) <= 5 ? 8 : 9;  // Timber litter
            }

            // Create some non-burnable areas (water, rock)
            if (elev < base_elevation - elevation_range * 0.4f) {
                cell.fuel_model = 0;  // Water
            }

            // Fuel moisture varies with aspect (to be calculated) and random
            cell.fuel_moisture = moisture_dist(rng);

            // Canopy cover based on fuel model
            if (cell.fuel_model >= 8 && cell.fuel_model <= 10) {
                cell.canopy_cover = 0.4f + noise(rng) * 0.3f;
            }
            else if (cell.fuel_model >= 4 && cell.fuel_model <= 7) {
                cell.canopy_cover = 0.2f + noise(rng) * 0.2f;
            }
            else {
                cell.canopy_cover = noise(rng) * 0.1f;
            }
            cell.canopy_cover = std::max(0.0f, std::min(1.0f, cell.canopy_cover));
        }
    }

    std::cout << "Terrain generation complete." << std::endl;
}

// Load elevation from GeoTIFF using GDAL
bool TerrainLoader::loadElevation(const std::string& filepath) {
    GDALAllRegister();

    GDALDataset* dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (dataset == nullptr) {
        std::cerr << "Failed to open elevation file: " << filepath << std::endl;
        return false;
    }

    width_ = dataset->GetRasterXSize();
    height_ = dataset->GetRasterYSize();

    std::cout << "Loading elevation: " << width_ << " x " << height_ << std::endl;

    // Get geotransform
    double geotransform[6];
    if (dataset->GetGeoTransform(geotransform) == CE_None) {
        origin_lon_ = geotransform[0];
        origin_lat_ = geotransform[3];
        pixel_scale_x_ = geotransform[1];
        pixel_scale_y_ = geotransform[5];
        // Calculate cell size in meters (approximate at this latitude)
        cell_size_ = std::abs(pixel_scale_x_) * 111320.0f * cosf(origin_lat_ * DEG_TO_RAD);
    }

    std::cout << "  Origin: " << origin_lon_ << ", " << origin_lat_ << std::endl;
    std::cout << "  Cell size: " << cell_size_ << " m" << std::endl;

    // Allocate memory
    terrain_.resize(width_ * height_);
    elevation_raw_.resize(width_ * height_);

    // Read elevation band
    GDALRasterBand* band = dataset->GetRasterBand(1);
    std::vector<float> buffer(width_ * height_);

    CPLErr err = band->RasterIO(GF_Read, 0, 0, width_, height_,
        buffer.data(), width_, height_,
        GDT_Float32, 0, 0);

    if (err != CE_None) {
        std::cerr << "Failed to read elevation data" << std::endl;
        GDALClose(dataset);
        return false;
    }

    // Get nodata value
    int hasNoData;
    double nodata = band->GetNoDataValue(&hasNoData);

    // Copy to terrain structure
    for (int i = 0; i < width_ * height_; ++i) {
        if (hasNoData && buffer[i] == nodata) {
            elevation_raw_[i] = 0.0f;
            terrain_[i].fuel_model = 0;  // Non-burnable
        }
        else {
            elevation_raw_[i] = buffer[i];
        }
        terrain_[i].elevation = elevation_raw_[i];
        terrain_[i].fuel_moisture = 0.05f;  // Default 5% moisture
        terrain_[i].canopy_cover = 0.0f;
    }

    GDALClose(dataset);
    std::cout << "Elevation loaded successfully" << std::endl;

    return true;
}

// Load fuel model from GeoTIFF using GDAL
bool TerrainLoader::loadFuelModel(const std::string& filepath) {
    GDALAllRegister();

    GDALDataset* dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (dataset == nullptr) {
        std::cerr << "Failed to open fuel model file: " << filepath << std::endl;
        return false;
    }

    int fuel_width = dataset->GetRasterXSize();
    int fuel_height = dataset->GetRasterYSize();

    std::cout << "Loading fuel model: " << fuel_width << " x " << fuel_height << std::endl;

    if (fuel_width != width_ || fuel_height != height_) {
        std::cerr << "Warning: Fuel dimensions (" << fuel_width << "x" << fuel_height
            << ") don't match elevation (" << width_ << "x" << height_ << ")" << std::endl;
    }

    // Read fuel model band
    GDALRasterBand* band = dataset->GetRasterBand(1);
    std::vector<int16_t> buffer(width_ * height_);

    band->RasterIO(GF_Read, 0, 0, width_, height_,
        buffer.data(), width_, height_,
        GDT_Int16, 0, 0);

    // Get nodata value
    int hasNoData;
    double nodata = band->GetNoDataValue(&hasNoData);

    // Count fuel types for debug
    int burnable = 0, nonburnable = 0;

    // LANDFIRE FBFM13 codes:
    // 1-13 = Anderson fuel models
    // 91=water, 92=snow, 93=agriculture, 98=urban, 99=barren
    // -9999 or other = nodata
    for (int i = 0; i < width_ * height_; ++i) {
        int16_t code = buffer[i];

        if (code >= 1 && code <= 13) {
            terrain_[i].fuel_model = static_cast<uint8_t>(code);
            burnable++;
        }
        else if (code == 91 || code == 92 || code == 93 || code == 98 || code == 99) {
            terrain_[i].fuel_model = 0;  // Non-burnable
            nonburnable++;
        }
        else if ((hasNoData && code == (int16_t)nodata) || code == -9999 || code == 0) {
            terrain_[i].fuel_model = 0;  // NoData = non-burnable
            nonburnable++;
        }
        else {
            // Unknown code, default to grass
            terrain_[i].fuel_model = 1;
            burnable++;
        }
    }

    GDALClose(dataset);

    std::cout << "Fuel model loaded: " << burnable << " burnable, "
        << nonburnable << " non-burnable cells" << std::endl;

    return true;
}

// Calculate slope and aspect from elevation
void TerrainLoader::calculateSlopeAspect() {
    if (width_ <= 2 || height_ <= 2) return;

    std::cout << "Calculating slope and aspect..." << std::endl;

    // Use 3x3 window with Horn's method
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            calcSlopeAspectAt(x, y);
        }
    }

    // Adjust fuel moisture based on aspect
    // North-facing slopes retain more moisture
    for (int i = 0; i < width_ * height_; ++i) {
        TerrainCell& cell = terrain_[i];
        if (cell.fuel_model == 0) continue;  // Skip non-burnable

        // South aspects are drier (better for fire spread)
        // Keep moisture LOW so fire can spread - typical fire weather is 3-8%
        float aspect_factor = cosf(cell.aspect * DEG_TO_RAD);
        cell.fuel_moisture = 0.03f + 0.04f * (1.0f + aspect_factor) * 0.5f;
        // Results in 3-7% moisture, well below extinction threshold
    }

    std::cout << "Slope/aspect calculation complete." << std::endl;
}

// Calculate slope and aspect at a specific cell using Horn's method
void TerrainLoader::calcSlopeAspectAt(int x, int y) {
    int idx = y * width_ + x;
    TerrainCell& cell = terrain_[idx];

    // Get 3x3 neighborhood elevations
    // Handle edges by clamping
    auto getElev = [this](int px, int py) -> float {
        px = std::max(0, std::min(width_ - 1, px));
        py = std::max(0, std::min(height_ - 1, py));
        return elevation_raw_[py * width_ + px];
        };

    // Horn's method uses weighted differences
    // z1 z2 z3
    // z4 z5 z6
    // z7 z8 z9
    float z1 = getElev(x - 1, y - 1);
    float z2 = getElev(x, y - 1);
    float z3 = getElev(x + 1, y - 1);
    float z4 = getElev(x - 1, y);
    // z5 = center (not needed)
    float z6 = getElev(x + 1, y);
    float z7 = getElev(x - 1, y + 1);
    float z8 = getElev(x, y + 1);
    float z9 = getElev(x + 1, y + 1);

    // dz/dx and dz/dy
    float dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0f * cell_size_);
    float dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0f * cell_size_);

    // Slope in degrees
    float slope_rad = atanf(sqrtf(dzdx * dzdx + dzdy * dzdy));
    cell.slope = slope_rad * RAD_TO_DEG;

    // Aspect (direction slope faces, in degrees from north)
    // Note: Aspect is direction of steepest descent (downhill)
    if (dzdx == 0.0f && dzdy == 0.0f) {
        cell.aspect = 0.0f;  // Flat
    }
    else {
        // atan2 gives angle from positive x-axis
        // We want angle from north (positive y), clockwise
        float aspect_rad = atan2f(dzdx, -dzdy);
        cell.aspect = aspect_rad * RAD_TO_DEG;
        if (cell.aspect < 0.0f) cell.aspect += 360.0f;
    }
}

// Load fuel moisture (stub)
bool TerrainLoader::loadFuelMoisture(const std::string& filepath) {
    std::cerr << "Fuel moisture loading not yet implemented." << std::endl;
    return false;
}

// Load canopy cover (stub)
bool TerrainLoader::loadCanopyCover(const std::string& filepath) {
    std::cerr << "Canopy cover loading not yet implemented." << std::endl;
    return false;
}

// Coordinate transforms (placeholders - would use actual georeferencing)
void TerrainLoader::getCellCoords(double lon, double lat, int& x, int& y) const {
    // Simple placeholder - assumes origin at (0,0)
    x = static_cast<int>((lon - origin_lon_) / pixel_scale_x_);
    y = static_cast<int>((lat - origin_lat_) / pixel_scale_y_);

    x = std::max(0, std::min(width_ - 1, x));
    y = std::max(0, std::min(height_ - 1, y));
}

void TerrainLoader::getGeoCoords(int x, int y, double& lon, double& lat) const {
    lon = origin_lon_ + x * pixel_scale_x_;
    lat = origin_lat_ + y * pixel_scale_y_;
}

// WindFieldLoader implementation
void WindFieldLoader::setUniform(float speed_mps, float direction_deg) {
    uniform_wind_.speed = speed_mps;
    uniform_wind_.direction = direction_deg;
    is_uniform_ = true;

    // Create single-element wind field
    wind_field_.clear();
    wind_field_.push_back(uniform_wind_);
}

bool WindFieldLoader::loadFromGrib(const std::string& filepath) {
    // TODO: Implement using eccodes or similar library
    std::cerr << "GRIB loading not yet implemented." << std::endl;
    return false;
}

void WindFieldLoader::interpolateToGrid(int width, int height, float cell_size) {
    if (is_uniform_) {
        // Just replicate uniform wind
        wind_field_.resize(width * height, uniform_wind_);
    }
    // TODO: Bilinear interpolation for spatially varying wind
}

// ValidationLoader implementation
bool ValidationLoader::loadFIRMS(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open FIRMS file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        FireDetection det;

        // FIRMS CSV format: latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,...
        // Parse latitude
        std::getline(ss, token, ',');
        det.latitude = std::stod(token);

        // Parse longitude
        std::getline(ss, token, ',');
        det.longitude = std::stod(token);

        // Parse brightness
        std::getline(ss, token, ',');
        det.brightness = std::stof(token);

        // Skip scan, track
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');

        // Parse acquisition date and time (simplified)
        std::getline(ss, token, ',');  // date
        std::string date = token;
        std::getline(ss, token, ',');  // time
        // TODO: Convert to Unix timestamp
        det.acq_time = 0.0;  // Placeholder

        // Parse satellite
        std::getline(ss, token, ',');
        det.satellite = token;

        // Confidence is later in the line
        det.confidence = 50.0f;  // Default

        detections_.push_back(det);
    }

    std::cout << "Loaded " << detections_.size() << " fire detections from FIRMS" << std::endl;
    return true;
}

std::vector<FireDetection> ValidationLoader::getDetectionsInRange(
    double start_time, double end_time) const {

    std::vector<FireDetection> result;
    for (const auto& det : detections_) {
        if (det.acq_time >= start_time && det.acq_time <= end_time) {
            result.push_back(det);
        }
    }
    return result;
}

std::vector<std::pair<int, int>> ValidationLoader::toGridCells(
    const TerrainLoader& terrain,
    double start_time, double end_time) const {

    auto dets = getDetectionsInRange(start_time, end_time);
    std::vector<std::pair<int, int>> cells;

    for (const auto& det : dets) {
        int x, y;
        terrain.getCellCoords(det.longitude, det.latitude, x, y);
        cells.emplace_back(x, y);
    }

    return cells;
}

float ValidationLoader::calculateJaccard(
    const SimCell* simulated,
    int width, int height,
    const TerrainLoader& terrain,
    double sim_start_time,
    double sim_end_time) const {

    // Get observed fire cells
    auto observed_cells = toGridCells(terrain, sim_start_time, sim_end_time);

    // Convert to set for efficient lookup
    std::vector<bool> observed_grid(width * height, false);
    for (const auto& cell : observed_cells) {
        int idx = cell.second * width + cell.first;
        if (idx >= 0 && idx < width * height) {
            observed_grid[idx] = true;
        }
    }

    // Calculate intersection and union
    int intersection = 0;
    int union_count = 0;

    for (int i = 0; i < width * height; ++i) {
        bool sim_burned = (simulated[i].state == BURNING || simulated[i].state == BURNED);
        bool obs_burned = observed_grid[i];

        if (sim_burned && obs_burned) {
            ++intersection;
        }
        if (sim_burned || obs_burned) {
            ++union_count;
        }
    }

    if (union_count == 0) return 1.0f;  // Both empty = perfect match

    return static_cast<float>(intersection) / union_count;
}