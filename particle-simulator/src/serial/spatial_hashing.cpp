#include "spatial_hashing.h"

int SpatialHash::hash(int x, int y) const {
    return y * 10000 + x;
}

SpatialHash::SpatialHash(float size) : cellSize(size) {}

void SpatialHash::insert(Particle* particle) {
    Vector position = particle->getPosition();
    int cellX = static_cast<int>(position.getX() / cellSize);
    int cellY = static_cast<int>(position.getY() / cellSize);
    grid[hash(cellX, cellY)].particles.push_back(particle);
}

std::vector<Particle*> SpatialHash::query(const Particle* particle) const {
    Vector position = particle->getPosition();
    int cellX = static_cast<int>(position.getX() / cellSize);
    int cellY = static_cast<int>(position.getY() / cellSize);
    std::vector<Particle*> neighbors;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            auto it = grid.find(hash(cellX + dx, cellY + dy));
            if (it != grid.end()) {
                neighbors.insert(neighbors.end(), it->second.particles.begin(), it->second.particles.end());
            }
        }
    }

    return neighbors;
}

void SpatialHash::clear() {
    grid.clear();
}