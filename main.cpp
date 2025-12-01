#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include "kernel.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

GLuint vbo;
std::vector<Vertex> host_buffer;

// Variables Globales
TerrainParams params;
float camAngleY = 0.0f;
float camZoom = -2.0f;
bool autoRotate = true;
float daySpeed = 0.5f; 

float3 lerpColor(float3 a, float3 b, float t) {
    t = std::max(0.0f, std::min(t, 1.0f)); 
    return { a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t };
}

void initGL() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, TOTAL_MAX_VERTS * sizeof(Vertex), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST); 
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Terreno Volumetrico Pro", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) return -1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    initCudaMemory();
    host_buffer.resize(TOTAL_MAX_VERTS);
    initGL();

    // Configuración Inicial (Restaurando valores por defecto)
    params.globalX = 0.0f;
    params.globalZ = 0.0f;
    params.waterLevel = -0.2f;
    params.scale = 1.0f;
    params.heightMult = 1.0f;
    params.octaves = 8;
    params.enableShadows = 1;
    params.isolevel = 0.0f;
    params.time = 3.14159f;

    float3 colNight  = {0.05f, 0.05f, 0.1f};
    float3 colSunset = {0.8f, 0.4f, 0.2f};
    float3 colDay    = {0.4f, 0.7f, 0.9f};  

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Ciclo Día/Noche
        params.time += daySpeed * 0.01f;
        float sunAngle = params.time;
        params.sunDir.x = cos(sunAngle); 
        params.sunDir.y = sin(sunAngle); 
        params.sunDir.z = 0.2f;

        float sunHeight = params.sunDir.y;
        float3 currentSky = (sunHeight > 0.0f) ? 
            lerpColor(colSunset, colDay, sunHeight * 1.5f) : 
            lerpColor(colSunset, colNight, -sunHeight * 3.0f);
            
        params.skyColor = currentSky;
        glClearColor(currentSky.x, currentSky.y, currentSky.z, 1.0f);

        // --- IMGUI ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controles Avanzados");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        
        if (ImGui::CollapsingHeader("Atmosfera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Velocidad Dia", &daySpeed, 0.0f, 2.0f);
            ImGui::Checkbox("Sombras Reales", (bool*)&params.enableShadows);
            ImGui::ColorEdit3("Cielo Dia", (float*)&colDay);
            ImGui::ColorEdit3("Cielo Noche", (float*)&colNight);
        }
        
        if (ImGui::CollapsingHeader("Terreno y Agua", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Nivel Agua", &params.waterLevel, -1.0f, 1.0f);
            ImGui::SliderFloat("Altura Montaña", &params.heightMult, 0.1f, 3.0f);
            ImGui::SliderFloat("Zoom (Escala)", &params.scale, 0.1f, 5.0f);
            ImGui::SliderInt("Detalle (Octavas)", &params.octaves, 1, 8);
            ImGui::SliderFloat("Grosor Terreno", &params.isolevel, -0.5f, 0.5f);
        }

        if (ImGui::CollapsingHeader("Navegacion", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Auto-Giro", &autoRotate);
            ImGui::SliderFloat("Zoom Camara", &camZoom, -10.0f, -0.1f);
        }
        ImGui::End();

        // Movimiento
        float speed = 0.05f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) params.globalZ += speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) params.globalZ -= speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) params.globalX -= speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) params.globalX += speed;
        if (autoRotate) camAngleY += 0.2f;

        // Render CUDA
        int numVerts = runCudaKernel(host_buffer.data(), params);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        if (numVerts > 0) glBufferSubData(GL_ARRAY_BUFFER, 0, numVerts * sizeof(Vertex), host_buffer.data());
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        float viewSize = -camZoom; 
        glOrtho(-viewSize, viewSize, -viewSize, viewSize, -10.0, 10.0); 
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glRotatef(30.0f, 1.0f, 0.0f, 0.0f); 
        glRotatef(camAngleY, 0.0f, 1.0f, 0.0f);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(4, GL_FLOAT, sizeof(Vertex), (void*)0);
        glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(4 * sizeof(float)));
        
        if (numVerts > 0) glDrawArrays(GL_TRIANGLES, 0, numVerts);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    cleanupCudaMemory();
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
    return 0;
}