#include "rothermel.cuh"

// Initialize Anderson 13 fuel models (host-only function)
__host__ void getAnderson13FuelModels(FuelModel* models) {
    // Model 0: Non-burnable (water, rock, etc.)
    models[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Model 1: Short grass (1 ft)
    models[1] = {
        0.166f,    // load: 0.74 tons/acre = 0.166 kg/m²
        0.305f,    // depth: 1.0 ft = 0.305 m
        0.12f,     // moisture of extinction
        18608.0f,  // heat content: 8000 BTU/lb = 18608 kJ/kg
        3500.0f,   // SAV: 3500 /ft = 11483 /m (using /ft for now)
        513.0f     // density: 32 lb/ft³ = 513 kg/m³
    };

    // Model 2: Timber grass/understory
    models[2] = {
        0.448f,    // load
        0.305f,    // depth
        0.15f,     // moisture ext
        18608.0f,  // heat
        3000.0f,   // SAV
        513.0f     // density
    };

    // Model 3: Tall grass (2.5 ft)
    models[3] = {
        0.674f,    // load
        0.762f,    // depth
        0.25f,     // moisture ext
        18608.0f,  // heat
        1500.0f,   // SAV
        513.0f     // density
    };

    // Model 4: Chaparral
    models[4] = {
        1.123f,    // load
        1.829f,    // depth
        0.20f,     // moisture ext
        18608.0f,  // heat
        2000.0f,   // SAV
        513.0f     // density
    };

    // Model 5: Brush (2 ft)
    models[5] = {
        0.224f,    // load
        0.610f,    // depth
        0.20f,     // moisture ext
        18608.0f,  // heat
        2000.0f,   // SAV
        513.0f     // density
    };

    // Model 6: Dormant brush
    models[6] = {
        0.336f,    // load
        0.762f,    // depth
        0.25f,     // moisture ext
        18608.0f,  // heat
        1750.0f,   // SAV
        513.0f     // density
    };

    // Model 7: Southern rough
    models[7] = {
        0.269f,    // load
        0.762f,    // depth
        0.40f,     // moisture ext
        18608.0f,  // heat
        1750.0f,   // SAV
        513.0f     // density
    };

    // Model 8: Compact timber litter
    models[8] = {
        0.337f,    // load
        0.061f,    // depth
        0.30f,     // moisture ext
        18608.0f,  // heat
        2000.0f,   // SAV
        513.0f     // density
    };

    // Model 9: Hardwood litter
    models[9] = {
        0.655f,    // load
        0.061f,    // depth
        0.25f,     // moisture ext
        18608.0f,  // heat
        2500.0f,   // SAV
        513.0f     // density
    };

    // Model 10: Timber understory
    models[10] = {
        0.897f,    // load
        0.305f,    // depth
        0.25f,     // moisture ext
        18608.0f,  // heat
        2000.0f,   // SAV
        513.0f     // density
    };

    // Model 11: Light slash
    models[11] = {
        0.337f,    // load
        0.305f,    // depth
        0.15f,     // moisture ext
        18608.0f,  // heat
        1500.0f,   // SAV
        513.0f     // density
    };

    // Model 12: Medium slash
    models[12] = {
        0.966f,    // load
        0.701f,    // depth
        0.20f,     // moisture ext
        18608.0f,  // heat
        1500.0f,   // SAV
        513.0f     // density
    };

    // Model 13: Heavy slash
    models[13] = {
        1.569f,    // load
        0.914f,    // depth
        0.25f,     // moisture ext
        18608.0f,  // heat
        1500.0f,   // SAV
        513.0f     // density
    };
}


