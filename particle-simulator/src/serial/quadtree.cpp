#include <vector>
#include <utility>

#include "quadtree.h"
#include "particle_serial.h"
#include "vector_serial.h"

Rectangle::Rectangle() {
}

Rectangle::Rectangle(float xb, float yb, float xt, float yt) {
    bottom = Vector(xb, yb);
    upper = Vector(xb + xt, yb + yt);
}

float Rectangle::getX() {
    return bottom.getX();
}

float Rectangle::getY(){
    return bottom.getY();
}

float Rectangle::getHeight(){
    return upper.getY() - bottom.getY();
}

float Rectangle::getWidth(){
    return upper.getX() - bottom.getX();
}

bool Rectangle::contains(Particle* p){


    bool x_interval = false;
    bool y_interval = false;

    if(bottom.getX() <= p->getVelocity().getX() and p->getVelocity().getX() <= upper.getX())
        x_interval = true;

    if(bottom.getY() <= p->getVelocity().getY() and p->getVelocity().getY() <= upper.getY())
        y_interval = true;

    return x_interval and y_interval;

}

Rectangle QuadTree::getBoundary(){
    return bounds;
}

std::vector<Particle*> QuadTree::getObjects(){
    return objs;
}

void QuadTree::insert(Particle* p){

    objs.push_back(p);

    // create child nodes
    if(nodes[0] == NULL and level <= MAX_LEVELS)
        split();


    for(int quadrant = 0; quadrant < 4; ++quadrant){
        if(nodes[quadrant] != NULL && nodes[quadrant]->getBoundary().contains(p)){
            nodes[quadrant]->insert(p);
        }
    }

}

QuadTree::QuadTree(int blevel, Rectangle b){
    level = blevel;
    bounds = b;

    for(int quadrant = 0; quadrant < 4; ++quadrant)
        nodes[quadrant] = NULL;
}


int QuadTree::getIndex(Particle* p){
    for(int i = 0; i<4; ++i)
        if(nodes[i]!= NULL && nodes[i]->bounds.contains(p))
            return i;

    return -1;
}

std::vector<Particle*> QuadTree::getQuadrant(int i)
{
    if (i >= 0 && i <= 3)
    {
        return nodes[i]->getObjects();
    }
    return getObjects();
}

std::vector<Particle*> QuadTree::retrieve(Particle* p){

    for(int quadrant = 0; quadrant < 4; ++quadrant){
        if(nodes[quadrant] != NULL && nodes[quadrant]->bounds.contains(p)){
            return nodes[quadrant]->retrieve(p);
        }
    }
    return objs;
}

void QuadTree::split(){
    float subWidth = (float)bounds.getWidth()/2.0; 
    float subHeight = (float)bounds.getHeight()/2.0; 

    float x = bounds.getX();
    float y = bounds.getY();


    nodes[0] = new QuadTree(level + 1, Rectangle(x + subWidth, y + subHeight, x + subWidth, y+subHeight));
    nodes[1] = new QuadTree(level + 1, Rectangle(x, y + subHeight, x+subWidth, y+subHeight));
    nodes[2] = new QuadTree(level + 1, Rectangle(x, y, x+subWidth, y+subHeight));
    nodes[3] = new QuadTree(level + 1, Rectangle(x + subWidth, y, x+subWidth, y+subHeight));
}

void QuadTree::clear(){

    objs.clear();

    for(int i = 0; i< 4; ++i){
        if(nodes[i] != NULL){
            nodes[i]->clear();
            nodes[i] = NULL;
        }
    }
}

////////// OLD CODE

// Quadtree::Quadtree() : x(0), y(0), width(0), height(0), level(0), maxLevel(0) {
//     // initialize children
//     std::vector<Particle>* children[4] = { new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>() };
// }

// Quadtree::Quadtree(float x, float y, float width, float height, int level, int maxLevel)
//     : x(x), y(y), width(width), height(height), level(level), maxLevel(maxLevel) {
//         // initialize children
//         std::vector<Particle>* children[4] = { new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>(), new std::vector<Particle>() };
//     }

// void Quadtree::checkCollisions(Particle p) {
//     int index = findIndex(p);
//     // invalid index
//     if (index == -1) {
//        return;
//     }

//     // only check for collisions for the appropriate section of the quadtree
//     for (Particle other : children[index]) {
//         // skip self-collision
//         if (p.getPosition().getX() == other.getPosition().getX() && 
//             p.getPosition().getY() == other.getPosition().getY()){
//             continue;
//         }

//         if (p.collidesWith(other)) {
//             p.resolveCollision(other);
//         }
//     }
// }

// // insert a particle into the quadtree according to its position
// void Quadtree::insert(Particle p) {

//     int index = findIndex(p);

//     if (index != -1) {
//         children[index].push_back(p);
//         return;
//     }
// }

// int Quadtree::findIndex(Particle p) {
//     float horizontalMidpoint = x + (width / 2);
//     float verticalMidpoint = y + (height / 2);

//     bool topQuadrant = p.getPosition().getY() >= verticalMidpoint;
//     bool leftQuadrant = p.getPosition().getX() <= horizontalMidpoint;

//     // find index of quadrant on the quadtree
//     if (leftQuadrant) {
//         if (topQuadrant) {
//             return 0;
//         } else {
//             return 2;
//         }
//     } else {
//         if (topQuadrant) {
//             return 1;
//         } else {
//             return 3;
//         }
//     }

//     // no valid index found
//     return -1;
// }

// void Quadtree::clear() {
//     children[0].clear();
//     children[1].clear();
//     children[2].clear();
//     children[3].clear();
// }

// std::vector<Particle> Quadtree::getParticles() {
//     std::vector<Particle> allParticles;

//     for (int i = 0; i < 4; i++) {
//         allParticles.insert(allParticles.end(), children[i].begin(), children[i].end());
//     }

//     return allParticles;
// }

// std::vector<Particle> Quadtree::getQuadrant(int index) {
//     return children[index];
// }


// void Quadtree::setX(float x) {
//     this->x = x;
// }

// void Quadtree::setY(float y) {
//     this->y = y;
// }

// void Quadtree::setWidth(float width) {
//     this->width = width;
// }

// void Quadtree::setHeight(float height) {
//     this->height = height;
// }

// void Quadtree::setLevel(int level) {
//     this->level = level;
// }

// void Quadtree::setMaxLevel(int maxLevel) {
//     this->maxLevel = maxLevel;
// }

// float Quadtree::getX() {
//     return x;
// }

// float Quadtree::getY() {
//     return y;
// }

// float Quadtree::getWidth() {
//     return width;
// }

// float Quadtree::getHeight() {
//     return height;
// }

// int Quadtree::getLevel() {
//     return level;
// }

// int Quadtree::getMaxLevel() {
//     return maxLevel;
// }

// //////// Failed attempt at splitting the quadtree into 4 quadrants ////////
// // void Quadtree::split() {
// //     float halfWidth = width / 2.0f;
// //     float halfHeight = height / 2.0f;

// //     children[0] = new Quadtree(x, y, halfWidth, halfHeight, level + 1, maxLevel);
// //     children[1] = new Quadtree(x + halfWidth, y, halfWidth, halfHeight, level + 1, maxLevel);
// //     children[2] = new Quadtree(x, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);
// //     children[3] = new Quadtree(x + halfWidth, y + halfHeight, halfWidth, halfHeight, level + 1, maxLevel);   
// // }
