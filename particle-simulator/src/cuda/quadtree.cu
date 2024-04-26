// #include <utility>

#include "quadtree.cuh"
#include "particle.cuh"
#include "vector.cuh"

Rectangle::Rectangle() {
}

// x and y for "b" = bottom left corner
// x and y for "t" = top right corner
Rectangle::Rectangle(float xb, float yb, float xt, float yt) :
    bottom(Vector::Vector(xb, yb)), upper(Vector::Vector(xt, yt)){}

__host__ __device__ float Rectangle::getX() {
    return bottom.getX();
}

__host__ __device__ float Rectangle::getY(){
    return bottom.getY();
}

__host__ __device__ float Rectangle::getHeight(){
    return upper.getY() - bottom.getY();
}

__host__ __device__ float Rectangle::getWidth(){
    return upper.getX() - bottom.getX();
}

__host__ __device__ bool Rectangle::contains(Particle* p){
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

__host__ __device__ Rectangle QuadTree::getBoundary(){
    return bounds;
}

__host__ __device__ std::vector<Particle*> QuadTree::getObjects(){
    return objs;
}

__host__ __device__ void QuadTree::insert(Particle* p){
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

__host__ __device__ void QuadTree::initLevels(){
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

__host__ __device__ int QuadTree::getIndex(Particle* p){
    for(int i = 0; i<4; ++i)
        if(nodes[i]!= NULL && nodes[i]->bounds.contains(p))
            return i;
    return -1;
}

__host__ __device__ std::vector<Particle*> QuadTree::getQuadrant(Particle* p)
{
    if (level >= MAX_LEVELS || nodes[0] == NULL) {
        return getObjects();
    }
    int index = getIndex(p);
    return nodes[index]->getQuadrant(p);
}

__host__ __device__ std::vector<Particle*> QuadTree::retrieve(Particle* p){
    for(int quadrant = 0; quadrant < 4; ++quadrant){
        if(nodes[quadrant] != NULL && nodes[quadrant]->bounds.contains(p)){
            return nodes[quadrant]->retrieve(p);
        }
    }
    return objs;
}

__host__ __device__ void QuadTree::split(){
    float subWidth = (float)bounds.getWidth()/2.0; 
    float subHeight = (float)bounds.getHeight()/2.0; 
    float x = bounds.getX();
    float y = bounds.getY();
    nodes[0] = new QuadTree(level + 1, Rectangle(x + subWidth, y + subHeight, x + subWidth, y+subHeight));
    nodes[1] = new QuadTree(level + 1, Rectangle(x, y + subHeight, x+subWidth, y+subHeight));
    nodes[2] = new QuadTree(level + 1, Rectangle(x, y, x+subWidth, y+subHeight));
    nodes[3] = new QuadTree(level + 1, Rectangle(x + subWidth, y, x+subWidth, y+subHeight));
}

__host__ __device__ void QuadTree::clear(){
    objs.clear();
    for(int i = 0; i< 4; ++i){
        if(nodes[i] != NULL){
            nodes[i]->clear();
            nodes[i] = NULL;
        }
    }
}