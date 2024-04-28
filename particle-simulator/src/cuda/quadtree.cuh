// #ifndef QUADTREE_CUH
// #define QUADTREE_CUH

// #include <cuda_runtime.h>
// #include "vector.cuh"
// #include "particle.cuh"

// class Rectangle {
// private:
//     Vector bottom;
//     Vector upper;

// public:
//     Rectangle(float xb, float yb, float xt, float yt);
//     Rectangle();
//     __host__ __device__ float getX();
//     __host__ __device__ float getY();
//     __host__ __device__ float getHeight();
//     __host__ __device__ float getWidth();
//     // must be __device__ //
//     __host__ __device__ bool contains(Particle* p);
// };

// class QuadTree {
// public:
//     QuadTree(int blevel, Rectangle b);
//     __host__ __device__ void insert(Particle* p);
//     __host__ __device__ void split();
//     __host__ __device__ void clear();
//     __host__ __device__ void initLevels();
//     __host__ __device__ Rectangle getBoundary();
//     // must use __device__ rectangle.contains() //
//     __host__ __device__ int getIndex(Particle* p);
//     // must be __device__ //
//     __device__ std::unordered_set<Particle*> getQuadrant(Particle* p);
//     __device__ Particle* quadTreeSetToArray();
//     __host__ __device__ std::unordered_set<Particle*> getObjects();

// private:
//     int MAX_OBJECTS = 4;
//     int MAX_LEVELS = 2;
//     int level;
//     Rectangle bounds;
//     std::unordered_set<Particle*> objs;
//     QuadTree* nodes[4];
// };

// #endif