#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include "kernel.h"

// ImGui
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

GLuint vbo, ibo, treeVbo;
std::vector<Vertex> host_buffer;
std::vector<unsigned int> indices;

// Variables Globales
TerrainParams params;
float camAngleY = 0.0f;
float camZoom = -1.8f;
bool autoRotate = true;
float daySpeed = 0.0f; 
bool wireframeMode = false;

// --- FUNCIÓN AUXILIAR PARA MEZCLAR COLORES (LERP) ---
float3 lerpColor(float3 a, float3 b, float t) {
    t = std::max(0.0f, std::min(t, 1.0f)); 
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}

void generateIndices() {
    indices.clear();
    indices.reserve(NUM_INDICES);
    
    // Generamos índices para formar quads con 2 triángulos cada uno
    for (unsigned int y = 0; y < MESH_HEIGHT - 1; y++) {
        for (unsigned int x = 0; x < MESH_WIDTH - 1; x++) {
            unsigned int topLeft = y * MESH_WIDTH + x;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (y + 1) * MESH_WIDTH + x;
            unsigned int bottomRight = bottomLeft + 1;
            
            // Primer triángulo (superior izquierdo)
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            
            // Segundo triángulo (inferior derecho)
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
}

void initGL() {
    // VBO para terreno
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    unsigned int size = NUM_TERRAIN_POINTS * sizeof(Vertex);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    
    // IBO para índices
    generateIndices();
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                 indices.data(), GL_STATIC_DRAW);
    
    // VBO para árboles (siguen siendo puntos)
    glGenBuffers(1, &treeVbo);
    glBindBuffer(GL_ARRAY_BUFFER, treeVbo);
    size = NUM_TERRAIN_POINTS * sizeof(Vertex);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST); 
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Terreno Pro - Malla de Triángulos", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) return -1;

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    initGL();
    initCudaMemory();
    host_buffer.resize(TOTAL_VERTICES);

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

    // Colores base
    float3 colNight  = {0.40f, 0.70f, 0.90f};
    float3 colSunset = {0.40f, 0.70f, 0.90f}; 
    float3 colDay    = {0.40f, 0.70f, 0.90f};

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        params.time += daySpeed * 0.01f;
        
        // Mover el sol
        float sunAngle = params.time;
        params.sunDir.x = cos(sunAngle); 
        params.sunDir.y = sin(sunAngle); 
        params.sunDir.z = 0.2f;

        float sunHeight = params.sunDir.y;
        float3 currentSky;

        if (sunHeight > 0.0f) {
            currentSky = lerpColor(colSunset, colDay, sunHeight * 1.5f);
        } else {
            currentSky = lerpColor(colSunset, colNight, -sunHeight * 3.0f);
        }

        params.skyColor = currentSky;
        glClearColor(currentSky.x, currentSky.y, currentSky.z, 1.0f);

        // INTERFAZ 
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Panel de Control");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        
        if (ImGui::CollapsingHeader("Atmosfera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Velocidad Dia", &daySpeed, 0.0f, 2.0f);
            
            float hour = fmod(params.time, 6.2831f) / 6.2831f * 24.0f;
            if (hour < 0) hour += 24.0f;
            ImGui::Text("Hora Virtual: %.1f h", hour);
            
            ImGui::Checkbox("Sombras Dinamicas", (bool*)&params.enableShadows);
            ImGui::ColorEdit3("Color Dia", (float*)&colDay);
            ImGui::ColorEdit3("Color Atardecer", (float*)&colSunset);
            ImGui::ColorEdit3("Color Noche", (float*)&colNight);
        }
        
        if (ImGui::CollapsingHeader("Terreno", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Nivel Agua", &params.waterLevel, -1.0f, 1.0f);
            ImGui::SliderFloat("Altura Montaña", &params.heightMult, 0.1f, 3.0f);
            ImGui::SliderFloat("Zoom (Escala)", &params.scale, 1.0f, 10.0f);
            ImGui::SliderInt("Detalle", &params.octaves, 1, 8);
        }

        if (ImGui::CollapsingHeader("Navegacion", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Auto-Giro", &autoRotate);
            ImGui::SliderFloat("Zoom Camara", &camZoom, -5.0f, -0.1f);
            ImGui::Checkbox("Modo Wireframe", &wireframeMode);
            if (ImGui::Button("Reestablecer Posicion")) { params.globalX = 0; params.globalZ = 0; }
        }
        ImGui::End();

        // MOVIMIENTO
        float speed = 0.03f * params.scale;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) params.globalZ -= speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) params.globalZ += speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) params.globalX -= speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) params.globalX += speed;
        if (autoRotate) camAngleY += 0.1f;

        // RENDER
        runCudaKernel(host_buffer.data(), params);

        // Actualizar VBO del terreno
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_TERRAIN_POINTS * sizeof(Vertex), host_buffer.data());
        
        // Actualizar VBO de árboles
        glBindBuffer(GL_ARRAY_BUFFER, treeVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_TERRAIN_POINTS * sizeof(Vertex), 
                       host_buffer.data() + NUM_TERRAIN_POINTS);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); 
        glLoadIdentity();
        float viewSize = -camZoom; 
        glOrtho(-viewSize, viewSize, -viewSize, viewSize, -10.0, 10.0); 
        
        glMatrixMode(GL_MODELVIEW); 
        glLoadIdentity();
        glRotatef(35.0f, 1.0f, 0.0f, 0.0f); 
        glRotatef(camAngleY, 0.0f, 1.0f, 0.0f);

        // Renderizar terreno con triángulos
        glPolygonMode(GL_FRONT_AND_BACK, wireframeMode ? GL_LINE : GL_FILL);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(4, GL_FLOAT, sizeof(Vertex), (void*)0);
        glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(4 * sizeof(float)));
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        
        // Renderizar árboles como puntos
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindBuffer(GL_ARRAY_BUFFER, treeVbo);
        glVertexPointer(4, GL_FLOAT, sizeof(Vertex), (void*)0);
        glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(4 * sizeof(float)));
        glPointSize(3.0f);
        glDrawArrays(GL_POINTS, 0, NUM_TERRAIN_POINTS);

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
    cleanupCudaMemory();
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &treeVbo);
    glfwTerminate();
    return 0;
}