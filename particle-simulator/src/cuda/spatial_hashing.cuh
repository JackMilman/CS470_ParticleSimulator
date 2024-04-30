#ifndef SPATIAL_HASHING_CUH
#define SPATIAL_HASHING_CUH

#include <thrust/device_vector.h>
#include "particle.cuh"
#include "vector.cuh"

class SpatialHash {
private:
    thrust::device_vector<int> keys;
    thrust::device_vector<int> particleIndices;
    float cellSize;

public:
    int storageSize;

    explicit SpatialHash(float size);
    void clear();
};

__global__ void insertParticles(Particle* particles, int numParticles, float cellSize, int* storageSize, int* keys, int* particleIndices);
__global__ void queryParticles(const Particle* particles, int numParticles, int* outputIndices, float cellSize, int* keys, int* particleIndices, int storageSize);

#endif