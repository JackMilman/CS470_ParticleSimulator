#include <vector>
#include <utility>

#include "quadtree.h"
#include "particle_serial.h"


Quadtree::Quadtree() : x(0), y(0), width(0), height(0), level(0), maxLevel(0) {
    // initialize children
    std::vector<Particle>* children[4] = { new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>() };
}

Quadtree::Quadtree(float x, float y, float width, float height, int level, int maxLevel)
    : x(x), y(y), width(width), height(height), level(level), maxLevel(maxLevel) {
        // initialize children
        std::vector<Particle>* children[4] = { new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>() };
    }

void Quadtree::checkCollisions(Particle p) {
    int index = findIndex(p);
    // invalid index
    if (index == -1) {
       return;
    }

    // only check for collisions for the appropriate section of the quadtree
    for (Particle other : children[index]) {
        // skip self-collision
        // if (p.getPosition().getX() == other.getPosition().getX() && 
        //     p.getPosition().getY() == other.getPosition().getY()){
        //     continue;
        // }

        if (p.collidesWith(other)) {
            p.resolveCollision(other);
        }
    }
}

// insert a particle into the quadtree according to its position
void Quadtree::insert(Particle p) {

    int index = findIndex(p);

    if (index != -1) {
        children[index].push_back(p);
        return;
    }
}

int Quadtree::findIndex(Particle p) {
    float horizontalMidpoint = x + (width / 2);
    float verticalMidpoint = y + (height / 2);

    bool topQuadrant = p.getPosition().getY() >= verticalMidpoint;
    bool leftQuadrant = p.getPosition().getX() <= horizontalMidpoint;

    // find index of quadrant on the quadtree
    if (leftQuadrant) {
        if (topQuadrant) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (topQuadrant) {
            return 1;
        } else {
            return 3;
        }
    }

    // no valid index found
    return -1;
}

void Quadtree::clear() {
    children[0].clear();
    children[1].clear();
    children[2].clear();
    children[3].clear();
}

std::vector<Particle> Quadtree::getParticles() {
    std::vector<Particle> allParticles;

    for (int i = 0; i < 4; i++) {
        allParticles.insert(allParticles.end(), children[i].begin(), children[i].end());
    }

    return allParticles;
}

std::vector<Particle> Quadtree::getQuadrant(int index) {
    return children[index];
}


void Quadtree::setX(float x) {
    this->x = x;
}

void Quadtree::setY(float y) {
    this->y = y;
}

void Quadtree::setWidth(float width) {
    this->width = width;
}

void Quadtree::setHeight(float height) {
    this->height = height;
}

void Quadtree::setLevel(int level) {
    this->level = level;
}

void Quadtree::setMaxLevel(int maxLevel) {
    this->maxLevel = maxLevel;
}

float Quadtree::getX() {
    return x;
}

float Quadtree::getY() {
    return y;
}

float Quadtree::getWidth() {
    return width;
}

float Quadtree::getHeight() {
    return height;
}

int Quadtree::getLevel() {
    return level;
}

int Quadtree::getMaxLevel() {
    return maxLevel;
}

//////// Failed attempt at splitting the quadtree into 4 quadrants ////////
// void Quadtree::split() {
//     float halfWidth = width / 2.0f;
//     float halfHeight = height / 2.0f;

//     children[0] = new Quadtree(x, y, halfWidth, halfHeight, level + 1, maxLevel);
//     children[1] = new Quadtree(x + halfWidth, y, halfWidth, halfHeight, level + 1, maxLevel);
//     children[2] = new Quadtree(x, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);
//     children[3] = new Quadtree(x + halfWidth, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);   
// }
