#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>       

#include "simulation.cuh"
#include "terrain.h"

int main(int argc, char** argv) {
    std::cout << "=== Wildfire Spread Simulator ===" << std::endl;
    std::cout << "CUDA-accelerated cellular automaton with Rothermel spread model" << std::endl;
    std::cout << std::endl;

    printDeviceInfo();
    return 0;
}