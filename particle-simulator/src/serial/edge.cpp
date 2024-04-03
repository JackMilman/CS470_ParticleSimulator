#include "particle_serial.h"
#include "edge.h"

// Default constructor
Edge::Edge() : parent(0), isLeft(false) {}
// Value constructor
Edge::Edge(int parent, bool isLeft) : parent(parent), isLeft(isLeft) {}

int Edge::getParentIdx() const {
    return parent;
}

bool Edge::getIsLeft() const {
    return isLeft;
}