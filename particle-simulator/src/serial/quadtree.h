#ifndef QUAD_TREE_H
#define QUAD_TREE_H
#include "vector_serial.h"
#include "particle_serial.h"

class Rectangle{
    private:

    Vector bottom;
    Vector upper;

    public:

    Rectangle(float xb, float yb, float xt, float yt);
    Rectangle();

    float getX();
    float getY();

    float getHeight();
    float getWidth();

    bool contains(Particle* p);
};

class QuadTree {
    private: 

    int MAX_OBJECTS = 4;
    int MAX_LEVELS = 2;

    int level;
    Rectangle bounds;

    std::vector<Particle*> objs;

    QuadTree* nodes[4];

    void split();

    public:

    std::vector<Particle*> getObjects();
    std::vector<Particle*> retrieve(Particle* p);

    QuadTree(int level, Rectangle bounds);
    int getIndex(Particle* p);
    // should return type be Particle?
    std::vector<Particle*> getQuadrant(int i);
    void clear();

    void insert(Particle* p);
    Rectangle getBoundary();
};
#endif // QUAD_TREE_H