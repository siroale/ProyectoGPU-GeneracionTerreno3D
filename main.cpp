#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include "kernel.h"

GLuint vbo;
std::vector<Vertex> host_buffer;

void initGL() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Tamaño total (terreno + árboles)
    unsigned int size = TOTAL_VERTICES * sizeof(Vertex);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glClearColor(0.5f, 0.7f, 0.9f, 1.0f);

    // IMPORTANTE: Habilitar mezcla (Blending) para que los puntos invisibles
    // de la capa de árboles no tapen el terreno.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Habilitar Depth Test para que las montañas tapen lo que está detrás
    glEnable(GL_DEPTH_TEST); 
}

int main() {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1024, 768, "Terreno 3D con Perlin Noise", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) return -1;

    initGL();
    initCudaMemory();
    // Reservar memoria en RAM para el TOTAL de vértices
    host_buffer.resize(TOTAL_VERTICES);

    float angle = 0.0f;
    float time = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        runCudaKernel(host_buffer.data(), time);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, host_buffer.size() * sizeof(Vertex), host_buffer.data());
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- AJUSTES DE CÁMARA ---
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        glOrtho(-1.5, 1.5, -1.5, 1.5, -10.0, 10.0); 
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glRotatef(35.0f, 1.0f, 0.0f, 0.0f); 

        // --- AJUSTE DE VELOCIDAD ---
        glRotatef(angle, 0.0f, 1.0f, 0.0f);
        angle += 0.1f; // Antes era 0.3f, ahora gira más lento
        time += 0.005f; // Avanzar en el tiempo genera mas terreno

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(4, GL_FLOAT, sizeof(Vertex), (void*)0);
        glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(4 * sizeof(float)));

        // Tamaño de los fucking puntos
        glPointSize(4.0f); 
        
        // DIBUJAR TODO (Terreno + Árboles)
        glDrawArrays(GL_POINTS, 0, NUM_TERRAIN_POINTS); // Usar TOTAL_VERTICES para que vuelvan los arboles feos

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanupCudaMemory();
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
    return 0;
}