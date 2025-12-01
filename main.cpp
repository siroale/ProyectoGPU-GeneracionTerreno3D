#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <chrono> // Para medir el tiempo

// Incluimos la version CPU en vez de la GPU
#include "cpu_kernel.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

GLuint vbo, ibo;
std::vector<Vertex> host_buffer;
std::vector<unsigned int> indices;

TerrainParams params;
float camAngleY = 0.0f;
float camZoom = -1.8f;
bool autoRotate = true;
float daySpeed = 0.0f; 
bool wireframeMode = false;

// Variable para guardar el tiempo de CPU
float cpuTimeMs = 0.0f;

void generateIndices() {
    indices.clear();
    indices.reserve(NUM_INDICES);
    
    for (unsigned int y = 0; y < MESH_HEIGHT - 1; y++) {
        for (unsigned int x = 0; x < MESH_WIDTH - 1; x++) {
            unsigned int topLeft = y * MESH_WIDTH + x;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (y + 1) * MESH_WIDTH + x;
            unsigned int bottomRight = bottomLeft + 1;
            
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
}

void initGL() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    unsigned int size = NUM_TERRAIN_POINTS * sizeof(Vertex);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    
    generateIndices();
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                 indices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST); 
    glEnable(GL_CULL_FACE);
}

int main() {
    if (!glfwInit()) return -1;
    // Cambiamos el título
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Terreno 3D - VERSION CPU (Lenta)", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Desactivar VSync para ver los FPS reales (bajos)
    if (glewInit() != GLEW_OK) return -1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    initGL();
    // Ya no hay initCudaMemory()
    
    host_buffer.resize(NUM_TERRAIN_POINTS);

    params.globalX = 0.0f;
    params.globalZ = 0.0f;
    
    params.waterLevel = -0.350f;
    params.scale = 2.133f;
    params.heightMult = 0.913f;
    params.octaves = 8;
    
    params.time = 3.14159f;
    daySpeed = 1.622f;
    params.enableShadows = 1;
    
    autoRotate = true;
    camZoom = -1.330f;

    float colNight[3]  = {0.40f, 0.70f, 0.90f};
    float colSunset[3] = {0.40f, 0.70f, 0.90f}; 
    float colDay[3]    = {0.40f, 0.70f, 0.90f};

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        params.time += daySpeed * 0.01f;
        
        float sunAngle = params.time;
        params.sunDir.x = cos(sunAngle); 
        params.sunDir.y = sin(sunAngle); 
        params.sunDir.z = 0.2f;

        float sunHeight = params.sunDir.y;
        
        // Lerp simple para el cielo en main
        float t = std::max(0.0f, std::min(sunHeight * 1.5f, 1.0f));
        if (sunHeight > 0.0f) {
             params.skyColor.x = colSunset[0] + (colDay[0] - colSunset[0]) * t;
             params.skyColor.y = colSunset[1] + (colDay[1] - colSunset[1]) * t;
             params.skyColor.z = colSunset[2] + (colDay[2] - colSunset[2]) * t;
        } else {
             t = std::max(0.0f, std::min(-sunHeight * 3.0f, 1.0f));
             params.skyColor.x = colSunset[0] + (colNight[0] - colSunset[0]) * t;
             params.skyColor.y = colSunset[1] + (colNight[1] - colSunset[1]) * t;
             params.skyColor.z = colSunset[2] + (colNight[2] - colSunset[2]) * t;
        }

        glClearColor(params.skyColor.x, params.skyColor.y, params.skyColor.z, 1.0f);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Benchmark CPU");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        
        // MOSTRAR TIEMPO DE CPU EN ROJO
        ImGui::TextColored(ImVec4(1,0,0,1), "Tiempo Generacion CPU: %.2f ms", cpuTimeMs);
        ImGui::Text("Puntos totales: %d", NUM_TERRAIN_POINTS);
        
        if (ImGui::CollapsingHeader("Atmosfera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Velocidad Dia", &daySpeed, 0.0f, 2.0f);
            ImGui::Checkbox("Sombras Dinamicas (Lento)", (bool*)&params.enableShadows);
        }
        
        if (ImGui::CollapsingHeader("Terreno", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Nivel Agua", &params.waterLevel, -1.0f, 1.0f);
            ImGui::SliderFloat("Altura Montana", &params.heightMult, 0.1f, 3.0f);
            ImGui::SliderFloat("Zoom (Escala)", &params.scale, 1.0f, 10.0f);
            ImGui::SliderInt("Detalle (Costoso)", &params.octaves, 1, 8);
        }

        if (ImGui::CollapsingHeader("Navegacion", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Auto-Giro", &autoRotate);
            ImGui::SliderFloat("Zoom Camara", &camZoom, -5.0f, -0.1f);
            ImGui::Checkbox("Modo Wireframe", &wireframeMode);
        }
        ImGui::End();

        float speed = 0.03f * params.scale;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) params.globalZ -= speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) params.globalZ += speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) params.globalX -= speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) params.globalX += speed;
        if (autoRotate) camAngleY += 0.1f;

        // --- BENCHMARK START ---
        auto start = std::chrono::high_resolution_clock::now();
        
        // Ejecutar Generación en CPU
        generateTerrainCPU(host_buffer, MESH_WIDTH, MESH_HEIGHT, params);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        cpuTimeMs = duration.count();
        // --- BENCHMARK END ---

        // Actualizar VBO del terreno
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_TERRAIN_POINTS * sizeof(Vertex), host_buffer.data());
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); 
        glLoadIdentity();
        float viewSize = -camZoom; 
        glOrtho(-viewSize, viewSize, -viewSize, viewSize, -10.0, 10.0); 
        
        glMatrixMode(GL_MODELVIEW); 
        glLoadIdentity();
        glRotatef(35.0f, 1.0f, 0.0f, 0.0f); 
        glRotatef(camAngleY, 0.0f, 1.0f, 0.0f);

        // Renderizar terreno
        glPolygonMode(GL_FRONT_AND_BACK, wireframeMode ? GL_LINE : GL_FILL);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(4, GL_FLOAT, sizeof(Vertex), (void*)0);
        glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(4 * sizeof(float)));
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    // cleanupCudaMemory() ya no es necesario
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    glfwTerminate();
    return 0;
}