#ifndef TERRAIN_H
#define TERRAIN_H

#include "simulation.cuh"
#include <string>
#include <vector>

// Terrain loading and processing utilities

class TerrainLoader {
public:
    TerrainLoader() = default;

    // Load elevation from GeoTIFF DEM
    bool loadElevation(const std::string& filepath);

    // Load fuel model grid from LANDFIRE FBFM
    bool loadFuelModel(const std::string& filepath);

    // Load fuel moisture from NFMD or computed values
    bool loadFuelMoisture(const std::string& filepath);

    // Load canopy cover from LANDFIRE
    bool loadCanopyCover(const std::string& filepath);

    // Generate synthetic terrain for testing
    void generateSynthetic(int width, int height,
        float base_elevation = 500.0f,
        float elevation_range = 200.0f);

    // Calculate slope and aspect from elevation
    void calculateSlopeAspect();

    // Get terrain data for upload to GPU
    const std::vector<TerrainCell>& getTerrain() const { return terrain_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

    // Get cell resolution
    float getCellSize() const { return cell_size_; }

    // Coordinate transforms
    void getCellCoords(double lon, double lat, int& x, int& y) const;
    void getGeoCoords(int x, int y, double& lon, double& lat) const;

private:
    std::vector<TerrainCell> terrain_;
    std::vector<float> elevation_raw_;  // raw elevation before slope/aspect calc

    int width_ = 0;
    int height_ = 0;
    float cell_size_ = 30.0f;  // meters

    // Georeference info
    double origin_lon_ = 0.0;
    double origin_lat_ = 0.0;
    double pixel_scale_x_ = 0.0;
    double pixel_scale_y_ = 0.0;

    // Helper for slope/aspect from 3x3 window
    void calcSlopeAspectAt(int x, int y);
};

// Wind field utilities
class WindFieldLoader {
public:
    // Load uniform wind
    void setUniform(float speed_mps, float direction_deg);

    // Load spatially varying wind from HRRR or similar
    bool loadFromGrib(const std::string& filepath);

    // Interpolate wind to simulation grid
    void interpolateToGrid(int width, int height, float cell_size);

    // Get wind data
    const std::vector<WindField>& getWindField() const { return wind_field_; }

    // For simple cases - get single wind value
    WindField getUniformWind() const { return uniform_wind_; }

private:
    std::vector<WindField> wind_field_;
    WindField uniform_wind_ = { 0.0f, 0.0f };
    bool is_uniform_ = true;
};

// Validation data loader (FIRMS satellite detections)
struct FireDetection {
    double longitude;
    double latitude;
    float brightness;       // Kelvin
    float confidence;       // 0-100
    double acq_time;        // Unix timestamp
    std::string satellite;  // "MODIS", "VIIRS", etc.
};

class ValidationLoader {
public:
    // Load FIRMS CSV data
    bool loadFIRMS(const std::string& filepath);

    // Filter detections by time range
    std::vector<FireDetection> getDetectionsInRange(
        double start_time, double end_time) const;

    // Convert to grid cells for comparison
    std::vector<std::pair<int, int>> toGridCells(
        const TerrainLoader& terrain,
        double start_time, double end_time) const;

    // Calculate Jaccard index (intersection/union) for validation
    float calculateJaccard(
        const SimCell* simulated,
        int width, int height,
        const TerrainLoader& terrain,
        double sim_start_time,
        double sim_end_time) const;

private:
    std::vector<FireDetection> detections_;
};

#endif // TERRAIN_H