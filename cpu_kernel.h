#pragma once

#include <vector>
#include <cmath>

#define MESH_WIDTH  512
#define MESH_HEIGHT 512
#define NUM_TERRAIN_POINTS (MESH_WIDTH * MESH_HEIGHT)
#define TOTAL_VERTICES (NUM_TERRAIN_POINTS * 2)

#define NUM_QUADS ((MESH_WIDTH - 1) * (MESH_HEIGHT - 1))
#define NUM_INDICES (NUM_QUADS * 6)

struct TerrainParams {
    float time;
    float globalX;
    float globalZ;
    float waterLevel;
    float scale;
    float heightMult;
    int   octaves;
    
    struct { float x, y, z; } sunDir;    
    struct { float x, y, z; } skyColor;  
    int   enableShadows; 
};

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

// Función principal que corre en CPU
void generateTerrainCPU(std::vector<Vertex>& vertices, int width, int height, TerrainParams p);