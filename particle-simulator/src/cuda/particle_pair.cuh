#ifndef PARTICLE_PAIR_H
#define PARTICLE_PAIR_H

#include <cuda_runtime.h>

class ParticlePair {
public:
    __host__ __device__ ParticlePair();
    __host__ __device__ ParticlePair(const int a, const int b);

    __host__ __device__ int getA() const;
    __host__ __device__ int getB() const;

private:
    int a;
    int b;
};

#endif // PARTICLE_PAIR_H