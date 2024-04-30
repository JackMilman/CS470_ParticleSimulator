#include "spatial_hashing.cuh"

SpatialHash::SpatialHash(float size) : cellSize(size), storageSize(0) {}

void SpatialHash::clear() {
    keys.clear();
    particleIndices.clear();
    storageSize = 0;
}

__global__ void insertParticles(Particle* particles, int numParticles, float cellSize, int* storageSize, int* keys, int* particleIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        Particle particle = particles[idx];
        Vector position = particle.getPosition();
        int cellX = static_cast<int>(position.getX() / cellSize);
        int cellY = static_cast<int>(position.getY() / cellSize);
        int hashed = cellY * 10000 + cellX;

        int storageIdx = atomicAdd(storageSize, 1);
        keys[storageIdx] = hashed;
        particleIndices[storageIdx] = idx;
    }
}

__global__ void queryParticles(Particle* particles, int numParticles, float cellSize, int* storageSize, int* keys, int* particleIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        Vector position = particles[idx].getPosition();
        int cellX = static_cast<int>(position.getX() / cellSize);
        int cellY = static_cast<int>(position.getY() / cellSize);
        int hashed = cellY * 10000 + cellX;

        for (int i = 0; i < *storageSize; ++i) {
            if (keys[i] == hashed) {
                int otherIdx = particleIndices[i];
                if (otherIdx != idx && particles[idx].collidesWith(particles[otherIdx])) {
                    particles[idx].resolveCollision(particles[otherIdx]);
                }
            }
        }
    }
}
