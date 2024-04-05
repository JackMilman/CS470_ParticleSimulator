#ifndef EDGE_H
#define EDGE_H
#include <cuda_runtime.h>

class Edge {
public:
    Edge();
    Edge(int parent, bool isLeft);
    
    __host__ int getParentIdx() const;
    __host__ bool getIsLeft() const;
private:
    int parent;
    bool isLeft;
};
#endif