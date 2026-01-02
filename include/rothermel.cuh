#ifndef ROTHERMEL_CUH
#define ROTHERMEL_CUH

#include <cuda_runtime.h>
#include "simulation.cuh"
#include <cmath>

// Rothermel spread model constants
namespace Rothermel {
    // Physical constants
    constexpr float HEAT_OF_PREIGNITION = 250.0f;  // kJ/kg (heat needed to ignite fuel)
    constexpr float AMBIENT_TEMP = 300.0f;         // Kelvin

    // Wind adjustment factors
    constexpr float MIDFLAME_HEIGHT_FACTOR = 0.4f; // 6.1m wind to midflame adjustment

    // Math constants
    constexpr float PI = 3.14159265358979323846f;
    constexpr float DEG_TO_RAD = PI / 180.0f;
    constexpr float RAD_TO_DEG = 180.0f / PI;
}

// ============================================================================
// All __device__ functions are defined inline in the header to avoid
// multiple definition errors when linking multiple .cu files
// ============================================================================

// Calculate moisture damping coefficient
// From Rothermel 1972, Eq. 29
__device__ __forceinline__ float moistureDamping(float moisture, float moisture_ext) {
    if (moisture >= moisture_ext) return 0.0f;

    float rm = moisture / moisture_ext;  // Moisture ratio

    // Polynomial damping function
    float eta_m = 1.0f - 2.59f * rm + 5.11f * rm * rm - 3.52f * rm * rm * rm;

    return fmaxf(0.0f, fminf(1.0f, eta_m));
}

// Calculate mineral damping coefficient  
// From Rothermel 1972, Eq. 30
__device__ __forceinline__ float mineralDamping(float mineral_content = 0.0555f) {
    // Default silica-free mineral content is 5.55%
    return 0.174f * powf(mineral_content, -0.19f);
}

// Calculate reaction intensity (IR)
// From Rothermel 1972, Eq. 27
__device__ __forceinline__ float reactionIntensity(
    const FuelModel& fuel,
    float moisture_damping
) {
    // Optimal packing ratio depends on SAV
    float sigma = fuel.sav_ratio;
    float beta_op = 3.348f * powf(sigma, -0.8189f);  // Eq. 37

    // Actual packing ratio
    float bulk_density = fuel.load / fuel.depth;
    float beta = bulk_density / fuel.density;

    // Relative packing ratio
    float beta_ratio = beta / beta_op;

    // Maximum reaction velocity (Eq. 36)
    float gamma_max = powf(sigma, 1.5f) / (495.0f + 0.0594f * powf(sigma, 1.5f));

    // Optimum reaction velocity (Eq. 38)
    float A = 133.0f * powf(sigma, -0.7913f);  // Eq. 39
    float gamma = gamma_max * powf(beta_ratio, A) * expf(A * (1.0f - beta_ratio));

    // Mineral damping
    float eta_s = mineralDamping();

    // Net fuel load (accounting for mineral content)
    float w_n = fuel.load * (1.0f - 0.0555f);  // Assume 5.55% total mineral

    // Reaction intensity (kW/m�)
    float IR = gamma * w_n * fuel.heat_content * moisture_damping * eta_s;

    return IR;
}

// Calculate propagating flux ratio (xi)
// From Rothermel 1972, Eq. 42
__device__ __forceinline__ float propagatingFluxRatio(float sav_ratio, float packing_ratio) {
    float sigma = sav_ratio;
    float beta = packing_ratio;

    float xi = expf((0.792f + 0.681f * sqrtf(sigma)) * (beta + 0.1f))
        / (192.0f + 0.2595f * sigma);

    return xi;
}

// Calculate heat sink (Qig * rho_b * epsilon)
// From Rothermel 1972, denominator of Eq. 52
__device__ __forceinline__ float heatSink(
    float bulk_density,
    float moisture,
    float heat_of_preignition
) {
    // Effective heating number (Eq. 14)
    float epsilon = 0.8f;  // Typical for fine fuels

    // Heat required for ignition (Eq. 12)
    // Qig = 250 + 1116*Mf (kJ/kg)
    float Qig = 250.0f + 1116.0f * moisture;

    return bulk_density * epsilon * Qig;
}

// Calculate slope factor (phi_s)
// From Rothermel 1972, Eq. 51
__device__ __forceinline__ float slopeFactor(float slope_degrees) {
    if (slope_degrees <= 0.0f) return 0.0f;

    // Convert to tan of slope
    float tan_slope = tanf(slope_degrees * Rothermel::DEG_TO_RAD);

    // Packing ratio dependency (simplified)
    float phi_s = 5.275f * tan_slope * tan_slope;

    return phi_s;
}

// Calculate wind factor (phi_w)
// From Rothermel 1972, Eq. 47-49
__device__ __forceinline__ float windFactor(float wind_speed, float sav_ratio, float fuel_depth) {
    if (wind_speed <= 0.0f) return 0.0f;

    float sigma = sav_ratio;

    // C, B, E coefficients (Eqs. 48-50)
    float C = 7.47f * expf(-0.133f * powf(sigma, 0.55f));
    float B = 0.02526f * powf(sigma, 0.54f);
    float E = 0.715f * expf(-3.59e-4f * sigma);

    // Packing ratio
    float bulk_density = 10.0f;  // kg/m�, typical
    float particle_density = 513.0f;  // kg/m�, typical for dead wood
    float beta = bulk_density / particle_density;

    // Wind coefficient
    float phi_w = C * powf(wind_speed, B) * powf(beta / 0.0012f, -E);

    // Limit maximum wind factor (physical constraint)
    phi_w = fminf(phi_w, 20.0f);

    return phi_w;
}

// Vector combination of wind and slope effects
__device__ __forceinline__ void vectorSpread(
    float base_rate,
    float phi_slope,
    float phi_wind,
    float slope_direction,  // upslope direction (degrees from north)
    float wind_direction,   // direction fire is pushed (degrees from north)
    float* max_rate,
    float* max_direction
) {
    // Convert directions to radians
    float slope_rad = slope_direction * Rothermel::DEG_TO_RAD;
    float wind_rad = wind_direction * Rothermel::DEG_TO_RAD;

    // Slope vector (points uphill)
    float slope_x = phi_slope * sinf(slope_rad);
    float slope_y = phi_slope * cosf(slope_rad);

    // Wind vector (points in direction wind pushes fire)
    float wind_x = phi_wind * sinf(wind_rad);
    float wind_y = phi_wind * cosf(wind_rad);

    // Combined vector
    float total_x = slope_x + wind_x;
    float total_y = slope_y + wind_y;

    // Magnitude of combined effect
    float phi_combined = sqrtf(total_x * total_x + total_y * total_y);

    // Direction of maximum spread
    float direction = atan2f(total_x, total_y) * Rothermel::RAD_TO_DEG;
    if (direction < 0.0f) direction += 360.0f;

    // Maximum spread rate
    *max_rate = base_rate * (1.0f + phi_combined);
    *max_direction = direction;
}

// Main spread rate calculation
__device__ __forceinline__ float calculateSpreadRate(
    const FuelModel& fuel,
    float fuel_moisture,
    float slope_degrees,
    float wind_speed,
    float wind_direction,  // direction wind comes FROM (meteorological convention)
    float aspect,          // direction slope faces
    float* out_direction
) {
    // Check for non-burnable conditions
    if (fuel.load <= 0.0f || fuel_moisture >= fuel.moisture_ext) {
        *out_direction = 0.0f;
        return 0.0f;
    }

    // Calculate moisture damping
    float eta_m = moistureDamping(fuel_moisture, fuel.moisture_ext);

    // Calculate reaction intensity
    float IR = reactionIntensity(fuel, eta_m);

    // Packing ratio
    float bulk_density = fuel.load / fuel.depth;
    float beta = bulk_density / fuel.density;

    // Propagating flux ratio
    float xi = propagatingFluxRatio(fuel.sav_ratio, beta);

    // Heat sink
    float hs = heatSink(bulk_density, fuel_moisture, Rothermel::HEAT_OF_PREIGNITION);

    // No-wind, no-slope spread rate (m/s)
    float R0 = (IR * xi) / (hs + 0.001f);  // Avoid div by zero

    // Convert from m/min to m/s
    R0 = R0 / 60.0f;

    // Calculate wind and slope factors
    float phi_s = slopeFactor(fabsf(slope_degrees));
    float phi_w = windFactor(wind_speed, fuel.sav_ratio, fuel.depth);

    // Upslope direction is opposite of aspect
    float upslope_dir = fmodf(aspect + 180.0f, 360.0f);

    // Wind pushes fire in direction opposite to where wind comes from
    float wind_push_dir = fmodf(wind_direction + 180.0f, 360.0f);

    // If slope is negative (downhill in spread direction), reduce slope factor
    if (slope_degrees < 0.0f) {
        phi_s *= 0.3f;
    }

    // Vector combination of wind and slope
    float max_rate, max_dir;
    vectorSpread(R0, phi_s, phi_w, upslope_dir, wind_push_dir, &max_rate, &max_dir);

    *out_direction = max_dir;

    // Physical limits on spread rate
    max_rate = fmaxf(0.0f, fminf(max_rate, 10.0f));  // Max 10 m/s is extreme

    return max_rate;
}

// Calculate fireline intensity (Byram's equation)
__device__ __forceinline__ float firelineIntensity(
    float spread_rate,
    float heat_per_area,
    float reaction_intensity
) {
    // Simplified using reaction intensity
    float flame_depth = spread_rate * 60.0f;  // Rough estimate
    float I = reaction_intensity * flame_depth;

    return I;
}

// Host-only function declarations
__host__ void getAnderson13FuelModels(FuelModel* models);

#endif // ROTHERMEL_CUH
