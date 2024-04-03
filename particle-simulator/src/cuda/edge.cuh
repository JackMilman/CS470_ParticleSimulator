#ifndef EDGE_H
#define EDGE_H
#include <cuda_runtime.h>

class Edge {
public:
    Edge();
    Edge(const Particle& parent, bool isLeft);
    
    __host__ const Particle& getParent() const;
    __host__ float getX() const;
    __host__ bool getIsLeft() const;
private:
    Particle parent;
    bool isLeft;
};
#endif