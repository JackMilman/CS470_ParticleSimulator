#ifndef QUADTREE_CUH
#define QUADTREE_CUH

#include <cuda_runtime.h>
#include vector.cuh
#include particle.cuh

class Rectangle {
private:
    Vector bottom;
    Vector upper;

public:
    Rectangle(float xb, float yb, float xt, float yt);
    Rectangle();
    __host__ __device__ float getX();
    __host__ __device__ float getY();
    __host__ __device__ float getHeight();
    __host__ __device__ float getWidth();
    __host__ __device__ bool contains(Particle* p);
};

class QuadTree {
public:
    // Constructors and methods
    QuadTree(int blevel, Rectangle b);
    // ~QuadTree();
    __host__ __device__ void insert(Particle* p);
    __host__ __device__ void split();
    __host__ __device__ void clear();
    __host__ __device__ void initLevels();
    __host__ __device__ Rectangle getBoundary();
    __host__ __device__ int getIndex(Particle* p);
    __host__ __device__ Particle** getQuadrant(Particle* p);
    __host__ __device__ Particle* getObjects();

private:
    int MAX_OBJECTS = 4;
    int MAX_LEVELS = 8;
    int level;
    Rectangle bounds;
    Particle* objs;
    QuadTree* nodes[4];
    // __host__ __device__ void split();
};
