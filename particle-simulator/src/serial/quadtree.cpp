#include <vector>
#include <utility>

#include "particle_serial.h"


class Quadtree {
public:
    Quadtree() : x(0), y(0), width(0), height(0), level(0), maxLevel(0) {}
    Quadtree(float x, float y, float width, float height, int level, int maxLevel)
        : x(x), y(y), width(width), height(height), level(level), maxLevel(maxLevel) {}
        
    void checkCollisions(Particle p) {
        // error check for empty quadtree
        if (children[0] == nullptr) {
            return;
        }

        // only check for collisions for the appropriate section of the quadtree
        for (int i = 0; i < 4; i++) {
            if (children[i]->contains(p)) {
                for (Particle other : children[i]->particles) {
                    
                    // skip self-collision
                    if (p.getPosition().getX() == other.getPosition().getX() && 
                        p.getPosition().getY() == other.getPosition().getY()){
                        continue;
                    }

                    if (p.collidesWith(other)) {
                        p.resolveCollision(other);
                    }
                }
                return;
            }
        }
    }

    // insert a particle into the quadtree according to its position
    void insert(Particle p) {
        if (children[0] == nullptr) {
            return;
        }

        if (level == maxLevel) {
            particles.push_back(p);
            return;
        }

        if (children[0] == nullptr) {
            split();
        }

        for (int i = 0; i < 4; i++) {
            if (children[i]->contains(p)) {
                children[i]->insert(p);
                return;
            }
        }

        // If we reach here, it means that the particle does not fit into any child, so we store it at this level.
        particles.push_back(p);
    }

    bool contains(Particle p) {
        float px = p.getPosition().getX();
        float py = p.getPosition().getY();
        return (px >= x && px <= x + width && py >= y && py <= y + height);
    }

    void clear() {
        particles.clear();

        for (int i = 0; i < 4; i++) {
            if (children[i] != nullptr) {
                children[i]->clear();
                delete children[i];
                children[i] = nullptr;
            }
        }
    }

    std::vector<Particle> getParticles() {
        return particles;
    }

private:
    void split() {
        float halfWidth = width / 2.0f;
        float halfHeight = height / 2.0f;

        children[0] = new Quadtree(x, y, halfWidth, halfHeight, level + 1, maxLevel);
        children[1] = new Quadtree(x + halfWidth, y, halfWidth, halfHeight, level + 1, maxLevel);
        children[2] = new Quadtree(x, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);
        children[3] = new Quadtree(x + halfWidth, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);
    }

    float x, y, width, height;
    int level, maxLevel;
    std::vector<Particle> particles;
    Quadtree* children[4] = { nullptr, nullptr, nullptr, nullptr };
};
