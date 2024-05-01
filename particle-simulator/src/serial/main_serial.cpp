#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <unistd.h>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <iomanip>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle_serial.h"
#include "particle_serial.cpp"
#include "vector_serial.h"
#include "vector_serial.cpp"
#include "edge.cpp"
#include "edge.h"
#include "quadtree.cpp"
// #include "quadtree.h"
#include "spatial_hashing.h"
#include "spatial_hashing.cpp"

#include <math.h>
#include <string.h>
#define DEFAULT_P_SIZE 0.05f
#define DEFAULT_P_NUMBER 50
#define PI 3.14159265f
#define NUM_CMD "-n num_particles"
#define SIZE_CMD "-s particle_size"
#define EXPLODE_CMD "-e explode_from_center"
#define SWEEP_CMD "-w sweep_and_prune"
#define QUAD_CMD "-t quad_tree"
#define SPATIAL_CMD "-g spatial_hash"
#define HELP_CMD "-h help"

int num_particles;
float particle_size;
Particle* particles;

enum modes {BruteForce, SweepAndPrune, Quad, Hash};
int mode = BruteForce;

Rectangle* rectangle;
QuadTree* quadtree;
Edge* edgesByX;
int num_edges;
SpatialHash* spatial_hash;
std::unordered_set<int>* p_overlaps;

int lastTime;

// Testing variables
std::chrono::duration<double, std::milli> cumulativeTime(0);

unsigned long long bruteForceOps = 0;
unsigned long long sweepAndPruneOps = 0;
unsigned long long spatialHashOps = 0;
unsigned long long treeOps = 0;

std::chrono::duration<double> bruteForceTime(0);
std::chrono::duration<double> sweepAndPruneTime(0);
std::chrono::duration<double> spatialHashTime(0);

// GL functionality
bool initGL(int *argc, char **argv);

void sortByX(Edge* edges) {
    // Simple insertion sort for AABB edges, sorting by their x-positions. This is to be used in sweep-and-prune.
    for (int i = 1; i < num_edges; i++) {
        for (int j = i - 1; j >= 0; j--) {
            Particle& p_j = particles[edges[j].getParentIdx()];
            Particle& p_next_j = particles[edges[j + 1].getParentIdx()];

            bool j_left = edges[j].getIsLeft();
            float j_x = j_left ? p_j.getPosition().getX() - particle_size: p_j.getPosition().getX() + particle_size;

            bool j_next_left = edges[j + 1].getIsLeft();
            float j_next_x = j_next_left ? p_next_j.getPosition().getX() - particle_size: p_next_j.getPosition().getX() + particle_size;

            if (j_x < j_next_x) break;
            Edge tmp = edges[j];
            edges[j] = edges[j + 1];
            edges[j + 1] = tmp;
        }
    }
}

// A simple check to determine if a particle pair has already been added to our overlap tracker.
bool resolved(int p_edge, int other) {
    bool resolved = p_overlaps[p_edge].count(other) == 1;
    return resolved;
}

/* Sweeps across the list of particle edges, sorted by their minimum x-values. 
   If an edge is a left-edge, we look at all the other particles currently
   being "touched" by our imaginary line and check if they have already been
   resolved. If they have not yet been resolved, we perform a finer-grained
   check to see if they collide, and resolve a collision if they do. */
int sweepAndPruneByX() {
    int num_ops = 0;
    sortByX(edgesByX);
    std::unordered_set<int> touching; // indexes of particles touched by the line at this point

    int p_edge_idx;
    int checked = 0;
    for (int i = 0; i < num_edges; i++) {
        p_edge_idx = edgesByX[i].getParentIdx();
        if (edgesByX[i].getIsLeft()) {
            for (auto itr = touching.begin(); itr != touching.end(); ++itr) {
                num_ops++;
                bool checked = resolved(p_edge_idx, *itr);
                if (!checked) {
                    if (particles[p_edge_idx].collidesWith(particles[*itr])) {
                        particles[p_edge_idx].resolveCollision(particles[*itr]);                      
                    }
                    p_overlaps[p_edge_idx].insert(*itr);
                    p_overlaps[*itr].insert(p_edge_idx);
                }
            }
            touching.insert(p_edge_idx);
        } else {
            touching.erase(p_edge_idx);
        }
    }
    // Resets the overlapping pairs sets for the next iteration of the algorithm.
    for (int i = 0; i < num_particles; i++) {
        p_overlaps[i].clear();
    }
    return num_ops;
}

// Updates all particles into the next time-step, only performing wall-bounce checking.
void stepParticles(float delta) {
    for (int i = 0; i < num_particles; i++) {
        particles[i].render();
        particles[i].updatePosition(delta);
        particles[i].wallBounce();
    }
}

// Compare one particle with all other particles to determine collisions
int bruteForceCheck(Particle& p) {
    int num_ops = 0;
    for (int j = 0; j < num_particles; j++) {
        if (p.collidesWith(particles[j])) {
            p.resolveCollision(particles[j]);
        }
        num_ops++;
    }
    return num_ops;
}

int quadTreeCheck(Particle& p) {
    int num_ops = 0;
    std::vector<Particle*> neighbors = quadtree->getQuadrant(&p);
    for (Particle* neighbor : neighbors) {
        // skip checking collision with self
        if (neighbor->getPosition().getX() == p.getPosition().getX() && 
            neighbor->getPosition().getY() == p.getPosition().getY()) {
            continue;
        } else {
            num_ops++;
            if (p.collidesWith(*neighbor)) {
                p.resolveCollision(*neighbor);
            }
        }
    }
    return num_ops;
}

int spatialHashCheck(Particle& p) {
    int num_ops = 0;
    auto neighbors = spatial_hash->query(&p);
    for (Particle* neighbor: neighbors) {
        num_ops++;
        if (&p != neighbor && p.collidesWith(*neighbor)) {
            p.resolveCollision(*neighbor);
        }
    }
    return num_ops;
}

// OpenGL rendering
void display() {
	glClear(GL_COLOR_BUFFER_BIT);
    
    // FPS counter
    static int frameCount = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    float delta = (currentTime - lastTime) / 1000.0f;
    lastTime = currentTime;
    frameCount++;

    if (frameCount == 1000) {
        double averageTime = cumulativeTime.count() / frameCount;
        std::cout << "Average time per frame: " 
              << std::fixed << std::setprecision(10) 
              << averageTime << " ms" << std::endl;
        switch (mode) {
            case BruteForce:
                std::cout << "Brute Force Ops: " << bruteForceOps << "\n";
                break;
            case SweepAndPrune:
                std::cout << "Sweep and Prune Ops: " << sweepAndPruneOps << "\n";
                break;
            case Quad:
                std::cout << "Quadtree Ops: " << treeOps << "\n";
                break;
            case Hash:
                std::cout << "Spatial Hash Ops: " << spatialHashOps << "\n";
                break;
            default:
                break;
        }
        exit(EXIT_SUCCESS);
    }

    if (frameCount % 20 == 0) {
        char title[80];
        sprintf(title, "Particle Simulator (%.2f fps) - %d particles", 1 / delta, num_particles);
        // printf("%f\n", 1 / delta);
        glutSetWindowTitle(title);
    }

    int num_ops = 0;
    stepParticles(delta);
    auto start = std::chrono::high_resolution_clock::now();
    switch (mode) {
        case BruteForce:
            for (int i = 0; i < num_particles; i++) {
                num_ops += bruteForceCheck(particles[i]);
            }
            break;
        case SweepAndPrune:
            num_ops += sweepAndPruneByX();
            break;
        case Quad:
            quadtree->clear();
            for (int i = 0; i < num_particles; i++) {
                quadtree->insert(&particles[i]);
            }
            for (int i = 0; i < num_particles; i++) {
                num_ops += quadTreeCheck(particles[i]);
            }
            break;
        case Hash:
            spatial_hash->clear();
            for (int i = 0; i < num_particles; i++) {
                spatial_hash->insert(&particles[i]);
            }
            for (int i = 0; i < num_particles; i++) {
                num_ops += spatialHashCheck(particles[i]);
            }            
            break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    cumulativeTime += end - start;
    switch (mode) {
        case BruteForce:
            bruteForceOps += num_ops;
            break;
        case SweepAndPrune:
            sweepAndPruneOps += num_ops;
            break;
        case Quad:
            treeOps += num_ops;
            break;
        case Hash:
            spatialHashOps += num_ops;
            break;
    }
    glutSwapBuffers();
}

void timer( int value )
{
    glutPostRedisplay();
    glutTimerFunc( 16, timer, 0 );
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitWindowSize(800, 800);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("Particle Simulator");
    glutPositionWindow(100,100);
    glutTimerFunc( 0, timer, 0 );
    glutDisplayFunc(display);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return false;
    }

    return true;
}

bool good_args(int argc, char** argv, bool* explode) {
     // Command line options
    int opt;
    while ((opt = getopt(argc, argv, "n:s:ewhtg")) != -1) {
        switch (opt) {
            case 'n':
                num_particles = strtol(optarg, NULL, 10);
                break;
            case 's':
                particle_size = strtod(optarg, NULL);
                break;
            case 'e':
                // Explode particles from center. Recommend running with a lot of particles with a low size
                *explode = true;
                break;
            case 'w':
                if (mode != BruteForce)
                    return false;
                mode = SweepAndPrune;
                break;
            case 't':
                if (mode != BruteForce)
                    return false;
                mode = Quad;
                break;
            case 'g':
                if (mode != BruteForce)
                    return false;
                mode = Hash;
                break;
            case 'h':
                return false;
                break;
            default:
                return false;
                break;
        }
        switch(mode) {
            case BruteForce:
                break;
            case SweepAndPrune:
                num_edges = num_particles * 2;
                edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));
                p_overlaps = new std::unordered_set<int>[num_particles];
                break;
            case Quad:
                rectangle = new Rectangle((float) X_MIN, (float) Y_MIN, (float) X_MAX, (float) Y_MAX);
                quadtree = new QuadTree(0, *rectangle);
                break;
            case Hash:
                spatial_hash = new SpatialHash(particle_size);
                break;
        }
    }
    return true;
}

int main(int argc, char** argv) {

    // Set defaults
    srand(time(NULL));

    num_particles = DEFAULT_P_NUMBER;
    particle_size = DEFAULT_P_SIZE;
    bool explode = false;
    if (!good_args(argc, argv, &explode)) {
        fprintf(stderr, "Usage: %s [%s] [%s] [%s (OPTIONAL)] [%s | %s | %s (OPTIONAL)]\n", argv[0],
            NUM_CMD, SIZE_CMD, EXPLODE_CMD, SWEEP_CMD, QUAD_CMD, SPATIAL_CMD);
        exit(EXIT_FAILURE);
    }
   

    particles = (Particle*) calloc(num_particles, sizeof(Particle));

    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Randomize velocity, position, and mass
        std::uniform_real_distribution<float> velocity(VEL_MIN, VEL_MAX);
        std::uniform_real_distribution<float> pos_x(X_MIN + particle_size, X_MAX - particle_size);
        std::uniform_real_distribution<float> pos_y(Y_MIN + particle_size, Y_MAX - particle_size);
        std::uniform_real_distribution<float> mass(1.5, 5);

        // make random particle velocity        
        float dx = velocity(gen);
        float dy = velocity(gen);
        // set particle positions
        float x, y;
        if (explode) {
            x = (X_MAX + X_MIN) / 2;
            y = (Y_MAX + Y_MIN) / 2;
        } else {
            x = pos_x(gen);
            y = pos_y(gen);
        }
        // create new particle with the randomized positions, mass, and with the set particle_size
        particles[i] = Particle(Vector(x, y), Vector(dx, dy), mass(gen), particle_size);
    }
    switch (mode) {
        case SweepAndPrune:
            // Initialize the list of edges, then sort them to prime the list for near-O(n) sorts.
            for (int i = 0; i < num_particles; i++) {
                edgesByX[i*2] = Edge(i, false);
                edgesByX[i*2 + 1] = Edge(i, true);
            }
            sortByX(edgesByX);
            break;
    }

    initGL(&argc, argv);
    lastTime = 0;
    glutMainLoop();

    return EXIT_SUCCESS;
}