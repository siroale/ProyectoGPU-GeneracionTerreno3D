#pragma once

#include <cuda_runtime.h>

#define MESH_WIDTH  512
#define MESH_HEIGHT 512
#define NUM_TERRAIN_POINTS (MESH_WIDTH * MESH_HEIGHT)
#define TOTAL_VERTICES (NUM_TERRAIN_POINTS * 2)

struct TerrainParams {
    float time;
    float globalX;
    float globalZ;
    float waterLevel;
    float scale;
    float heightMult;
    int   octaves;
    
    // Parámetros de iluminación
    float3 sunDir;    
    float3 skyColor;  
    int   enableShadows; 
};

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

void initCudaMemory();
void cleanupCudaMemory();
void runCudaKernel(Vertex* host_ptr, TerrainParams params);