#include "particle_pair.cuh"

// Default Constructor
__host__ __device__ ParticlePair::ParticlePair() : a(0), b(1) {}
__host__ __device__ ParticlePair::ParticlePair(const int a, const int b) : a(a), b(b) {}

__host__ __device__ int ParticlePair::getA() const {
    return a;
}

__host__ __device__ int ParticlePair::getB() const {
    return b;
}