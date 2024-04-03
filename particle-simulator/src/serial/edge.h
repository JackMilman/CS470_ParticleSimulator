#ifndef EDGE_H
#define EDGE_H

class Edge {
public:
    Edge();
    Edge(int parent, bool isLeft);
    
    int getParentIdx() const;
    // float getX() const;
    bool getIsLeft() const;
private:
    int parent;
    bool isLeft;
};
#endif