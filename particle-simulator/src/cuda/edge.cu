#include "particle.cuh"
#include "edge.cuh"

// Default constructor
Edge::Edge() : parent(0), isLeft(false) {}
// Value constructor
Edge::Edge(int parent, bool isLeft) : parent(parent), isLeft(isLeft) {}

__host__ int Edge::getParentIdx() const {
    return parent;
}

__host__ bool Edge::getIsLeft() const {
    return isLeft;
}