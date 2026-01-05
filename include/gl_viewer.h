#ifndef GL_VIEWER_H
#define GL_VIEWER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "simulation.cuh"
#include "terrain.h"
#include <vector>
#include <string>

class GLViewer {
public:
    GLViewer(int width, int height, const std::string& title);
    ~GLViewer();

    // Initialize with terrain data
    bool init(const TerrainLoader& terrain);

    // Update fire state (call each frame or when state changes)
    void updateFireState(const SimCell* cells, int num_cells);

    // Main render loop - returns false when window should close
    bool render();

    // Check if window should close
    bool shouldClose() const;

    // Get window for input handling
    GLFWwindow* getWindow() { return window_; }

    // Camera controls
    void handleInput(float dt);

    // Simulation control
    bool isPaused() const { return paused_; }
    void togglePause() { paused_ = !paused_; }
    float getSimSpeed() const { return sim_speed_; }
    void setSimSpeed(float speed) { sim_speed_ = speed; }

private:
    GLFWwindow* window_ = nullptr;
    int window_width_;
    int window_height_;

    // Terrain mesh
    GLuint terrain_vao_ = 0;
    GLuint terrain_vbo_ = 0;
    GLuint terrain_ebo_ = 0;
    GLuint terrain_color_vbo_ = 0;
    int num_indices_ = 0;

    // Fire overlay
    GLuint fire_vbo_ = 0;
    std::vector<float> fire_colors_;

    // Shaders
    GLuint shader_program_ = 0;

    // Camera
    glm::vec3 camera_pos_;
    glm::vec3 camera_front_;
    glm::vec3 camera_up_;
    float yaw_ = -90.0f;
    float pitch_ = -45.0f;
    float camera_speed_ = 500.0f;
    float mouse_sensitivity_ = 0.1f;
    double last_mouse_x_ = 0;
    double last_mouse_y_ = 0;
    bool first_mouse_ = true;
    bool mouse_captured_ = false;

    // Terrain info
    int grid_width_ = 0;
    int grid_height_ = 0;
    float cell_size_ = 30.0f;
    float min_elevation_ = 0.0f;
    float max_elevation_ = 1000.0f;

    // Simulation control
    bool paused_ = false;
    float sim_speed_ = 1.0f;

    // Helper functions
    bool createShaders();
    void createTerrainMesh(const TerrainLoader& terrain);
    void updateColors(const SimCell* cells);
    glm::vec3 getTerrainColor(float elevation, uint8_t fuel_model);
    glm::vec3 getFireColor(CellState state, float residence_time);

    // Callbacks
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};

#endif // GL_VIEWER_H