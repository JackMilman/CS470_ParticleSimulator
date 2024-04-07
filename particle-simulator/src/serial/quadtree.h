class Quadtree {
    public:
        Quadtree();
        Quadtree(float x, float y, float width, float height, int level, int maxLevel);
        void insert(Particle p);
        void checkCollisions(Particle p);
        void clear();
        std::vector<Particle> getParticles();
        int findIndex(Particle p);

        void setX(float x);
        void setY(float y);
        void setWidth(float width);
        void setHeight(float height);
        void setLevel(int level);
        void setMaxLevel(int maxLevel);

        float getX();
        float getY();
        float getWidth();
        float getHeight();
        int getLevel();
        int getMaxLevel();
        std::vector<Particle> getQuadrant(int index);

    private:
        float x;
        float y;
        float width;
        float height;
        int level;
        int maxLevel;
        std::vector<Particle> children[4];
};