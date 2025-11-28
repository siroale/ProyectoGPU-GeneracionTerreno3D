#ifndef KERNEL_H
#define KERNEL_H

#define MESH_WIDTH 512
#define MESH_HEIGHT 512

// Cantidad de puntos en la malla del suelo
#define NUM_TERRAIN_POINTS (MESH_WIDTH * MESH_HEIGHT)

// Total de vértices: Capa de suelo + Capa de árboles
#define TOTAL_VERTICES (NUM_TERRAIN_POINTS * 2)

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

void initCudaMemory();
void runCudaKernel(Vertex* host_ptr, float time);
void cleanupCudaMemory();

#endif