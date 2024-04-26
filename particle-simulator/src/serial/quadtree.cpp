#include <vector>
#include <utility>

#include "quadtree.h"
#include "particle_serial.h"
#include "vector_serial.h"

Rectangle::Rectangle() {
}

// x and y for "b" = bottom left corner
// x and y for "t" = top right corner
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
    if(bottom.getX() <= p->getVelocity().getX() and p->getVelocity().getX() <= upper.getX()) {
        x_interval = true;
        printf("true\n");
    }
    if(bottom.getY() <= p->getVelocity().getY() and p->getVelocity().getY() <= upper.getY()) {
        y_interval = true;
        printf("true\n");
    }
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
    initLevels();
}

void QuadTree::initLevels(){
    if (level < MAX_LEVELS){
        if (nodes[0] == NULL) {
            split();
        } 
        for (int i = 0; i < 4; i++) {
            nodes[i]->split();
            nodes[i]->initLevels();
        }
    }
}

int QuadTree::getIndex(Particle* p){
    for(int i = 0; i<4; ++i)
        if(nodes[i]!= NULL && nodes[i]->bounds.contains(p))
            return i;
    return -1;
}

std::vector<Particle*> QuadTree::getQuadrant(Particle* p)
{
    if (level >= MAX_LEVELS || nodes[0] == NULL) {
        return getObjects();
    }
    int index = getIndex(p);
    return nodes[index]->getQuadrant(p);
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