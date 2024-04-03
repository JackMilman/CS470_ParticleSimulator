#include "particle.cuh"
#include "edge.cuh"

// Default constructor
Edge::Edge() : parent(Particle()), isLeft(false) {}
// Value constructor
Edge::Edge(const Particle& parent, bool isLeft) : parent(parent), isLeft(isLeft) {}

__host__ const Particle& Edge::getParent() const {
    return parent;
}

__host__ float Edge::getX() const {
    if (isLeft) {
        return parent.getPosition().getX() - parent.getRadius();
    } else {
        return parent.getPosition().getX() + parent.getRadius();
    }
    
}

__host__ bool Edge::getIsLeft() const {
    return isLeft;
}