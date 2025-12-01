#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "kernel.h"
#include "tables.h" 

Vertex* d_vertices = nullptr;
int* d_vertexCount = nullptr;

// --- MATEMÁTICAS ---
__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }
__device__ float clamp(float v, float min, float max) { return fmaxf(min, fminf(v, max)); }
__device__ float fract(float x) { return x - floorf(x); }
__device__ float smoothstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// Hash rápido
__device__ float hash(float n) { return fract(sinf(n) * 43758.5453f); }

// Ruido 3D Suave
__device__ float noise3D(float3 x) {
    float3 p = {floorf(x.x), floorf(x.y), floorf(x.z)};
    float3 f = {fract(x.x), fract(x.y), fract(x.z)};
    // Curva de suavizado (Quintic) para eliminar artefactos lineales
    f.x = f.x*f.x*f.x*(f.x*(f.x*6.0f-15.0f)+10.0f);
    f.y = f.y*f.y*f.y*(f.y*(f.y*6.0f-15.0f)+10.0f);
    f.z = f.z*f.z*f.z*(f.z*(f.z*6.0f-15.0f)+10.0f);
    
    float n = p.x + p.y*57.0f + 113.0f*p.z;
    return lerp(lerp(lerp(hash(n+0.0f), hash(n+1.0f), f.x),
                     lerp(hash(n+57.0f), hash(n+58.0f), f.x), f.y),
                lerp(lerp(hash(n+113.0f), hash(n+114.0f), f.x),
                     lerp(hash(n+170.0f), hash(n+171.0f), f.x), f.y), f.z);
}

// Ruido 2D para montañas base (Ridge)
__device__ float ridge_noise(float x, float z, int octaves) {
    float total = 0.0f; float amp = 1.0f; float freq = 0.8f; float maxV = 0.0f;
    for(int i=0; i<octaves; i++) { 
        float n = 1.0f - fabsf(noise3D({x*freq, 0.0f, z*freq})*2.0f - 1.0f);
        // Elevamos al cuadrado para picos más afilados pero bases suaves
        n = n * n; 
        total += n * amp; 
        maxV += amp; 
        amp *= 0.5f; 
        freq *= 2.0f; 
    }
    return total/maxV;
}

// --- MAPA DE DENSIDAD (El "Cerebro" de la forma) ---
__device__ float map(float3 p, TerrainParams params) {
    // Escala global
    float3 pos = {p.x * params.scale, p.y * params.scale, p.z * params.scale};
    
    // 1. Terreno Base (Montañas)
    float h = ridge_noise(pos.x, pos.z, params.octaves) * params.heightMult;
    
    // Densidad inicial: Diferencia entre altura Y y el terreno
    // Usamos una transición suave en lugar de resta directa para bordes más redondos
    float density = p.y - (h - 0.5f); 

    // 2. Cuevas "Orgánicas"
    // Usamos ruido de baja frecuencia para túneles grandes
    float caveNoise = noise3D({pos.x * 0.8f, pos.y * 0.8f, pos.z * 0.8f});
    
    // smoothstep crea una transición suave en las paredes de la cueva
    // Solo excavamos si el ruido es muy alto (> 0.75)
    float hole = smoothstep(0.75f, 0.95f, caveNoise);
    
    // Sumar densidad "aire" donde hay cueva
    density += hole * 5.0f; 
    
    return density;
}

// --- NUEVO: CÁLCULO DE NORMALES SUAVES (Gradient) ---
// Esto elimina el look "low poly". Calcula hacia dónde apunta la superficie REAL.
__device__ float3 calculateNormal(float3 p, TerrainParams params) {
    float eps = 0.005f * params.scale; // Epsilon pequeño
    // Derivada parcial en los 3 ejes
    float dx = map({p.x + eps, p.y, p.z}, params) - map({p.x - eps, p.y, p.z}, params);
    float dy = map({p.x, p.y + eps, p.z}, params) - map({p.x, p.y - eps, p.z}, params);
    float dz = map({p.x, p.y, p.z + eps}, params) - map({p.x, p.y, p.z - eps}, params);
    
    float3 n = {dx, dy, dz};
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if(len > 0.0f) { n.x/=len; n.y/=len; n.z/=len; }
    return n; // Normal suavizada
}

// --- SOMBRAS VOLUMÉTRICAS ---
__device__ float calculateShadow(float3 pos, float3 sunDir, TerrainParams p) {
    if (p.enableShadows == 0 || sunDir.y < 0) return 1.0f;
    float shadow = 1.0f;
    float t = 0.02f; 
    float stepSize = 0.03f; // Pasos más cortos para mayor precisión

    for (int i = 0; i < 12; i++) {
        float3 rayPos = {pos.x + sunDir.x * t, pos.y + sunDir.y * t, pos.z + sunDir.z * t};
        if (rayPos.y > p.heightMult + 0.5f) break;

        float d = map(rayPos, p);
        // Soft shadows: Penumbra basada en cuán cerca pasamos del terreno
        if (d < p.isolevel) {
            shadow = 0.1f; 
            break;
        }
        t += stepSize;
    }
    return shadow;
}

__device__ float3 vertexInterp(float isolevel, float3 p1, float3 p2, float valp1, float valp2) {
    if (abs(isolevel - valp1) < 0.00001f) return p1;
    if (abs(isolevel - valp2) < 0.00001f) return p2;
    if (abs(valp1 - valp2) < 0.00001f) return p1;
    float mu = (isolevel - valp1) / (valp2 - valp1);
    return {p1.x + mu * (p2.x - p1.x), p1.y + mu * (p2.y - p1.y), p1.z + mu * (p2.z - p1.z)};
}

__global__ void marching_cubes_kernel(Vertex* vertices, int* vertexCount, TerrainParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= GRID_SIZE - 1 || y >= GRID_SIZE - 1 || z >= GRID_SIZE - 1) return;

    float step = 1.0f / (float)GRID_SIZE;
    float3 pos = {x * step, y * step, z * step};
    
    float3 globalPos = {
        pos.x * 2.0f - 1.0f + p.globalX, 
        pos.y * 2.0f - 1.0f, 
        pos.z * 2.0f - 1.0f + p.globalZ
    };

    // 1. Muestreo
    float cubeVal[8];
    float3 offsets[8] = {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}, {0,1,0}, {1,1,0}, {1,1,1}, {0,1,1}};
    float3 cornerPos[8];
    int cubeIndex = 0;

    for(int i=0; i<8; i++) {
        cornerPos[i] = {
            globalPos.x + offsets[i].x * step * 2.0f,
            globalPos.y + offsets[i].y * step * 2.0f,
            globalPos.z + offsets[i].z * step * 2.0f
        };
        cubeVal[i] = map(cornerPos[i], p);
        if (cubeVal[i] < p.isolevel) cubeIndex |= (1 << i);
    }

    int edges = edgeTable[cubeIndex];
    if (edges == 0) return;

    float3 vertList[12];
    // Interpolación de vértices (Código estándar MC)
    if (edges & 1) vertList[0] = vertexInterp(p.isolevel, cornerPos[0], cornerPos[1], cubeVal[0], cubeVal[1]);
    if (edges & 2) vertList[1] = vertexInterp(p.isolevel, cornerPos[1], cornerPos[2], cubeVal[1], cubeVal[2]);
    if (edges & 4) vertList[2] = vertexInterp(p.isolevel, cornerPos[2], cornerPos[3], cubeVal[2], cubeVal[3]);
    if (edges & 8) vertList[3] = vertexInterp(p.isolevel, cornerPos[3], cornerPos[0], cubeVal[3], cubeVal[0]);
    if (edges & 16) vertList[4] = vertexInterp(p.isolevel, cornerPos[4], cornerPos[5], cubeVal[4], cubeVal[5]);
    if (edges & 32) vertList[5] = vertexInterp(p.isolevel, cornerPos[5], cornerPos[6], cubeVal[5], cubeVal[6]);
    if (edges & 64) vertList[6] = vertexInterp(p.isolevel, cornerPos[6], cornerPos[7], cubeVal[6], cubeVal[7]);
    if (edges & 128) vertList[7] = vertexInterp(p.isolevel, cornerPos[7], cornerPos[4], cubeVal[7], cubeVal[4]);
    if (edges & 256) vertList[8] = vertexInterp(p.isolevel, cornerPos[0], cornerPos[4], cubeVal[0], cubeVal[4]);
    if (edges & 512) vertList[9] = vertexInterp(p.isolevel, cornerPos[1], cornerPos[5], cubeVal[1], cubeVal[5]);
    if (edges & 1024) vertList[10] = vertexInterp(p.isolevel, cornerPos[2], cornerPos[6], cubeVal[2], cubeVal[6]);
    if (edges & 2048) vertList[11] = vertexInterp(p.isolevel, cornerPos[3], cornerPos[7], cubeVal[3], cubeVal[7]);

    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
        int idx = atomicAdd(vertexCount, 3);
        if (idx >= TOTAL_MAX_VERTS - 3) break; 

        // Generar 3 vértices para el triángulo
        for(int k=0; k<3; k++) {
            float3 vPos = vertList[triTable[cubeIndex][i+k]]; 
            
            // --- AQUÍ ESTÁ LA MEJORA VISUAL CLAVE ---
            // Calculamos la normal EXACTA para este punto específico
            float3 norm = calculateNormal(vPos, p);
            
            // Iluminación difusa (Suave)
            float diff = fmaxf(norm.x*p.sunDir.x + norm.y*p.sunDir.y + norm.z*p.sunDir.z, 0.0f);
            
            // Sombra suave
            float shadow = calculateShadow(vPos, p.sunDir, p);
            float light = 0.25f + (diff * 0.75f * shadow); // Iluminación más balanceada

            // --- BIOMAS MEJORADOS ---
            float r, g, b;
            
            // Pendiente (Slope): Si la normal apunta hacia arriba, es plano. Si apunta a los lados, es pared.
            float slope = 1.0f - norm.y; 

            // AGUA (Transparencia simulada con color)
            if (vPos.y < p.waterLevel) {
                r = 0.0f; g = 0.3f; b = 0.7f; 
                light = 0.9f; // El agua brilla más
            } 
            else {
                // Biomas mezclados suavemente
                float hRel = (vPos.y - p.waterLevel) / p.heightMult;
                
                if (hRel < 0.08f) { // Playa
                    r = 0.76f; g = 0.70f; b = 0.50f; 
                } else if (hRel > 0.65f) { // Nieve
                    r = 0.95f; g = 0.95f; b = 1.0f;
                    // Roca si la pendiente es muy fuerte (nieve no se pega a paredes verticales)
                    if (slope > 0.6f) { r=0.4f; g=0.4f; b=0.4f; }
                } else { 
                    // Bosque / Hierba
                    r = 0.2f; g = 0.5f; b = 0.2f;
                    
                    // Si es muy inclinado, es roca de montaña
                    if (slope > 0.4f) {
                        float rockNoise = noise3D({vPos.x*5.0f, vPos.y*5.0f, vPos.z*5.0f});
                        r = 0.4f + rockNoise*0.1f; g = 0.35f + rockNoise*0.1f; b = 0.3f + rockNoise*0.1f;
                    }
                }
            }
            
            r *= light; g *= light; b *= light;

            // Niebla atmosférica
            float dist = sqrtf(globalPos.x*globalPos.x + globalPos.z*globalPos.z);
            float fog = clamp((dist - 1.2f) * 0.8f, 0.0f, 1.0f);
            r = lerp(r, p.skyColor.x, fog);
            g = lerp(g, p.skyColor.y, fog);
            b = lerp(b, p.skyColor.z, fog);

            vertices[idx+k] = {vPos.x, vPos.y, vPos.z, 1.0f, r, g, b, 1.0f};
        }
    }
}

// Host functions (Sin cambios grandes)
void initCudaMemory() {
    cudaMalloc((void**)&d_vertices, TOTAL_MAX_VERTS * sizeof(Vertex));
    cudaMalloc((void**)&d_vertexCount, sizeof(int));
}
void cleanupCudaMemory() {
    cudaFree(d_vertices);
    cudaFree(d_vertexCount);
}
int runCudaKernel(Vertex* host_ptr, TerrainParams params) {
    int zero = 0;
    cudaMemcpy(d_vertexCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Grid 3D ajustado
    dim3 b(8, 8, 8);
    dim3 g((GRID_SIZE + b.x - 1) / b.x, (GRID_SIZE + b.y - 1) / b.y, (GRID_SIZE + b.z - 1) / b.z);
    
    marching_cubes_kernel<<<g, b>>>(d_vertices, d_vertexCount, params);
    
    int count = 0;
    cudaMemcpy(&count, d_vertexCount, sizeof(int), cudaMemcpyDeviceToHost);
    if(count > 0) cudaMemcpy(host_ptr, d_vertices, count * sizeof(Vertex), cudaMemcpyDeviceToHost);
    return count;
}