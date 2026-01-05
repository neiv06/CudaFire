#include "gl_viewer.h"
#include <iostream>
#include <cmath>

// Vertex shader
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)";

// Fragment shader
const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

GLViewer::GLViewer(int width, int height, const std::string& title)
    : window_width_(width), window_height_(height) {

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // Anti-aliasing

    // Create window
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);

    // Set callbacks
    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
    glfwSetCursorPosCallback(window_, mouseCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    glfwSetKeyCallback(window_, keyCallback);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return;
    }

    // Enable depth testing and anti-aliasing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // Set clear color (sky blue)
    glClearColor(0.53f, 0.81f, 0.92f, 1.0f);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
}

GLViewer::~GLViewer() {
    if (terrain_vao_) glDeleteVertexArrays(1, &terrain_vao_);
    if (terrain_vbo_) glDeleteBuffers(1, &terrain_vbo_);
    if (terrain_ebo_) glDeleteBuffers(1, &terrain_ebo_);
    if (terrain_color_vbo_) glDeleteBuffers(1, &terrain_color_vbo_);
    if (shader_program_) glDeleteProgram(shader_program_);

    glfwTerminate();
}

bool GLViewer::init(const TerrainLoader& terrain) {
    grid_width_ = terrain.getWidth();
    grid_height_ = terrain.getHeight();
    cell_size_ = terrain.getCellSize();

    // Create shaders
    if (!createShaders()) {
        return false;
    }

    // Create terrain mesh
    createTerrainMesh(terrain);

    // Initialize camera position (above terrain center, looking down)
    float center_x = grid_width_ * cell_size_ * 0.5f;
    float center_y = grid_height_ * cell_size_ * 0.5f;
    float height = (max_elevation_ - min_elevation_) * 3.0f + 2000.0f;

    camera_pos_ = glm::vec3(center_x, height, center_y + height * 0.5f);
    camera_front_ = glm::normalize(glm::vec3(0.0f, -0.7f, -0.7f));
    camera_up_ = glm::vec3(0.0f, 1.0f, 0.0f);

    return true;
}

bool GLViewer::createShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        return false;
    }

    // Link program
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertexShader);
    glAttachShader(shader_program_, fragmentShader);
    glLinkProgram(shader_program_);

    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program_, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return true;
}

void GLViewer::createTerrainMesh(const TerrainLoader& terrain) {
    const auto& terrain_data = terrain.getTerrain();

    // Find elevation range
    min_elevation_ = 1e10f;
    max_elevation_ = -1e10f;
    for (const auto& cell : terrain_data) {
        if (cell.fuel_model > 0) {
            min_elevation_ = std::min(min_elevation_, cell.elevation);
            max_elevation_ = std::max(max_elevation_, cell.elevation);
        }
    }

    // Create vertices (position + color)
    std::vector<float> vertices;
    std::vector<float> colors;
    vertices.reserve(grid_width_ * grid_height_ * 3);
    colors.reserve(grid_width_ * grid_height_ * 3);
    fire_colors_.resize(grid_width_ * grid_height_ * 3);

    for (int z = 0; z < grid_height_; ++z) {
        for (int x = 0; x < grid_width_; ++x) {
            int idx = z * grid_width_ + x;
            const TerrainCell& cell = terrain_data[idx];

            // Position (x, y=elevation, z)
            vertices.push_back(x * cell_size_);
            vertices.push_back(cell.elevation);
            vertices.push_back(z * cell_size_);

            // Color based on terrain
            glm::vec3 color = getTerrainColor(cell.elevation, cell.fuel_model);
            colors.push_back(color.r);
            colors.push_back(color.g);
            colors.push_back(color.b);

            // Initialize fire colors to terrain colors
            fire_colors_[idx * 3 + 0] = color.r;
            fire_colors_[idx * 3 + 1] = color.g;
            fire_colors_[idx * 3 + 2] = color.b;
        }
    }

    // Create indices for triangle strips
    std::vector<unsigned int> indices;
    for (int z = 0; z < grid_height_ - 1; ++z) {
        for (int x = 0; x < grid_width_ - 1; ++x) {
            int topLeft = z * grid_width_ + x;
            int topRight = topLeft + 1;
            int bottomLeft = (z + 1) * grid_width_ + x;
            int bottomRight = bottomLeft + 1;

            // Two triangles per quad
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
    num_indices_ = static_cast<int>(indices.size());

    // Create VAO
    glGenVertexArrays(1, &terrain_vao_);
    glBindVertexArray(terrain_vao_);

    // Position VBO
    glGenBuffers(1, &terrain_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
        vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color VBO (dynamic - will be updated with fire state)
    glGenBuffers(1, &terrain_color_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, terrain_color_vbo_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float),
        colors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Index buffer
    glGenBuffers(1, &terrain_ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
        indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);

    std::cout << "Created terrain mesh: " << grid_width_ << "x" << grid_height_
        << " (" << num_indices_ / 3 << " triangles)" << std::endl;
}

glm::vec3 GLViewer::getTerrainColor(float elevation, uint8_t fuel_model) {
    if (fuel_model == 0) {
        return glm::vec3(0.1f, 0.1f, 0.4f);  // Water/non-burnable
    }

    float t = (elevation - min_elevation_) / (max_elevation_ - min_elevation_);
    t = glm::clamp(t, 0.0f, 1.0f);

    // Gradient: green -> olive -> brown
    glm::vec3 low(0.13f, 0.55f, 0.13f);   // Forest green
    glm::vec3 mid(0.42f, 0.56f, 0.14f);   // Olive
    glm::vec3 high(0.55f, 0.47f, 0.40f);  // Brown

    if (t < 0.5f) {
        return glm::mix(low, mid, t * 2.0f);
    }
    else {
        return glm::mix(mid, high, (t - 0.5f) * 2.0f);
    }
}

glm::vec3 GLViewer::getFireColor(CellState state, float residence_time) {
    switch (state) {
    case BURNING: {
        float t = glm::clamp(residence_time / 300.0f, 0.0f, 1.0f);
        if (t < 0.5f) {
            return glm::mix(glm::vec3(1.0f, 1.0f, 0.0f),  // Yellow
                glm::vec3(1.0f, 0.5f, 0.0f),   // Orange
                t * 2.0f);
        }
        else {
            return glm::mix(glm::vec3(1.0f, 0.5f, 0.0f),  // Orange
                glm::vec3(1.0f, 0.0f, 0.0f),   // Red
                (t - 0.5f) * 2.0f);
        }
    }
    case BURNED:
        return glm::vec3(0.15f, 0.15f, 0.15f);  // Dark gray
    default:
        return glm::vec3(0.0f);
    }
}

void GLViewer::updateFireState(const SimCell* cells, int num_cells) {
    for (int i = 0; i < num_cells; ++i) {
        glm::vec3 color;

        if (cells[i].state == BURNING) {
            color = getFireColor(BURNING, cells[i].residence_time);
        }
        else if (cells[i].state == BURNED) {
            color = getFireColor(BURNED, 0);
        }
        else {
            // Keep original terrain color - skip update
            continue;
        }

        fire_colors_[i * 3 + 0] = color.r;
        fire_colors_[i * 3 + 1] = color.g;
        fire_colors_[i * 3 + 2] = color.b;
    }

    // Update GPU buffer
    glBindBuffer(GL_ARRAY_BUFFER, terrain_color_vbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, fire_colors_.size() * sizeof(float),
        fire_colors_.data());
}

bool GLViewer::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void GLViewer::handleInput(float dt) {
    if (!window_) return;

    float velocity = camera_speed_ * dt;

    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
        camera_pos_ += velocity * camera_front_;
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
        camera_pos_ -= velocity * camera_front_;
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
        camera_pos_ -= glm::normalize(glm::cross(camera_front_, camera_up_)) * velocity;
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
        camera_pos_ += glm::normalize(glm::cross(camera_front_, camera_up_)) * velocity;
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS)
        camera_pos_ += velocity * camera_up_;
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS)
        camera_pos_ -= velocity * camera_up_;

    // Speed adjustment
    if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera_speed_ = 2000.0f;
    else
        camera_speed_ = 500.0f;
}

bool GLViewer::render() {
    if (!window_ || shouldClose()) return false;

    glfwPollEvents();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program_);

    // Set up matrices
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
        static_cast<float>(window_width_) / window_height_, 10.0f, 100000.0f);

    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "model"), 1, GL_FALSE, &model[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "projection"), 1, GL_FALSE, &projection[0][0]);

    // Draw terrain
    glBindVertexArray(terrain_vao_);
    glDrawElements(GL_TRIANGLES, num_indices_, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glfwSwapBuffers(window_);
    return true;
}

// Callbacks
void GLViewer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    GLViewer* viewer = static_cast<GLViewer*>(glfwGetWindowUserPointer(window));
    if (viewer) {
        viewer->window_width_ = width;
        viewer->window_height_ = height;
    }
}

void GLViewer::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    GLViewer* viewer = static_cast<GLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer || !viewer->mouse_captured_) return;

    if (viewer->first_mouse_) {
        viewer->last_mouse_x_ = xpos;
        viewer->last_mouse_y_ = ypos;
        viewer->first_mouse_ = false;
    }

    float xoffset = static_cast<float>(xpos - viewer->last_mouse_x_) * viewer->mouse_sensitivity_;
    float yoffset = static_cast<float>(viewer->last_mouse_y_ - ypos) * viewer->mouse_sensitivity_;

    viewer->last_mouse_x_ = xpos;
    viewer->last_mouse_y_ = ypos;

    viewer->yaw_ += xoffset;
    viewer->pitch_ += yoffset;
    viewer->pitch_ = glm::clamp(viewer->pitch_, -89.0f, 89.0f);

    glm::vec3 front;
    front.x = cos(glm::radians(viewer->yaw_)) * cos(glm::radians(viewer->pitch_));
    front.y = sin(glm::radians(viewer->pitch_));
    front.z = sin(glm::radians(viewer->yaw_)) * cos(glm::radians(viewer->pitch_));
    viewer->camera_front_ = glm::normalize(front);
}

void GLViewer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    GLViewer* viewer = static_cast<GLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    viewer->camera_speed_ += static_cast<float>(yoffset) * 100.0f;
    viewer->camera_speed_ = glm::clamp(viewer->camera_speed_, 100.0f, 5000.0f);
}

void GLViewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    GLViewer* viewer = static_cast<GLViewer*>(glfwGetWindowUserPointer(window));
    if (!viewer) return;

    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_SPACE:
            viewer->togglePause();
            std::cout << (viewer->isPaused() ? "Paused" : "Resumed") << std::endl;
            break;
        case GLFW_KEY_TAB:
            viewer->mouse_captured_ = !viewer->mouse_captured_;
            glfwSetInputMode(window, GLFW_CURSOR,
                viewer->mouse_captured_ ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            viewer->first_mouse_ = true;
            break;
        case GLFW_KEY_EQUAL:  // + key
            viewer->sim_speed_ *= 2.0f;
            std::cout << "Sim speed: " << viewer->sim_speed_ << "x" << std::endl;
            break;
        case GLFW_KEY_MINUS:
            viewer->sim_speed_ /= 2.0f;
            std::cout << "Sim speed: " << viewer->sim_speed_ << "x" << std::endl;
            break;
        }
    }
}