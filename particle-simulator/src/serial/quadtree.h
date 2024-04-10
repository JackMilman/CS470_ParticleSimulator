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




// class Quadtree {    
//     public:
//         Quadtree();
//         Quadtree(float x, float y, float width, float height, int level, int maxLevel);
//         void insert(Particle p);
//         void checkCollisions(Particle p);
//         void clear();
//         std::vector<Particle> getParticles();
//         int findIndex(Particle p);

//         void setX(float x);
//         void setY(float y);
//         void setWidth(float width);
//         void setHeight(float height);
//         void setLevel(int level);
//         void setMaxLevel(int maxLevel);

//         float getX();
//         float getY();
//         float getWidth();
//         float getHeight();
//         int getLevel();
//         int getMaxLevel();
//         std::vector<Particle> getQuadrant(int index);

//     private:
//         float x;
//         float y;
//         float width;
//         float height;
//         int level;
//         int maxLevel;
//         std::vector<Particle> children[4];
// };