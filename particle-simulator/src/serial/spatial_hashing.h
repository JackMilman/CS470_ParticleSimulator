#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H

#include <unordered_map>
#include <vector>
#include "particle_serial.h"
#include "vector_serial.h"

class SpatialHash {
private:
    struct Cell {
        std::vector<Particle*> particles;
    };
    std::unordered_map<int, Cell> grid;
    float cellSize;

    int hash(int x, int y) const;

public:
    explicit SpatialHash(float size);
    void insert(Particle* particle);
    std::vector<Particle*> query(const Particle* particle) const;
    void clear();
};

#endif // SPATIAL_HASH_H
