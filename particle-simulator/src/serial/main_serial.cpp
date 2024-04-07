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
#include "spatial_hashing.h"
#include "spatial_hashing.cpp"

#include <math.h>
#include <string.h>
#define DEFAULT_P_SIZE 0.05f
#define DEFAULT_P_NUMBER 50
#define PI 3.14159265f

int num_particles;
float particle_size;
Particle* particles;
Quadtree quadtree;

Edge* edgesByX;
int num_edges;
bool withSweep;
bool withTree;
bool withSpatialHash;
bool print_fps;
SpatialHash spatialHash(DEFAULT_P_SIZE);
std::unordered_set<int>* p_overlaps;

int lastTime;

// Testing variables
std::chrono::duration<double, std::milli> cumulativeTime(0);

unsigned long long bruteForceOps = 0;
unsigned long long sweepAndPruneOps = 0;
unsigned long long spatialHashOps = 0;

std::chrono::duration<double> bruteForceTime(0);
std::chrono::duration<double> sweepAndPruneTime(0);
std::chrono::duration<double> spatialHashTime(0);

// GL functionality
bool initGL(int *argc, char **argv);

void sortByX(Edge* edges) {
    // Simple insertion sort for the particles, sorting by their x-positions. This is to be used in sweep-and-prune.
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
void sweepAndPruneByX(int& num_ops) {
    sortByX(edgesByX);
    std::unordered_set<int> touching; // indexes of particles touched by the line at this point
    int p_edge_idx;
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
        if (withSweep) {
            std::cout << "Sweep and Prune Ops: " << sweepAndPruneOps << "\n";
        } else if (withSpatialHash) {
            std::cout << "Spatial Hash Ops: " << spatialHashOps << "\n";
        } else if (withTree) {

        } else {
            std::cout << "Brute Force Ops: " << bruteForceOps << "\n";
        }
        exit(0);
    }

    if (frameCount % 20 == 0) {
        char title[80];
        sprintf(title, "Particle Simulator (%.2f fps) - %d particles", 1 / delta, num_particles);
        if (print_fps) printf("%f\n", 1 / delta);
        glutSetWindowTitle(title);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;

    if (withSweep) {
        int num_ops = 0;
        for (int i = 0; i < num_particles; i++) {
            // Render the particle
            particles[i].render();
            // Update the particle's position, check for wall collision
            particles[i].updatePosition(delta);
            particles[i].wallBounce();
        }
        start = std::chrono::high_resolution_clock::now();
        // Sweep and prune algorithm
        sweepAndPruneByX(num_ops);
        end = std::chrono::high_resolution_clock::now();
        cumulativeTime += end - start;
        sweepAndPruneOps += num_ops;
    } else if (withSpatialHash) {
        int num_ops = 0;
        start = std::chrono::high_resolution_clock::now();
        spatialHash.clear();
        for (int i = 0; i < num_particles; i++) {
            particles[i].render();
            particles[i].updatePosition(delta);
            particles[i].wallBounce();
            spatialHash.insert(&particles[i]);
        }
        for (int i = 0; i < num_particles; i++) {
            Particle& particle = particles[i];
            auto neighbors = spatialHash.query(&particle);
            for (Particle* neighbor : neighbors) {
                num_ops++;
                if (&particle != neighbor && particle.collidesWith(*neighbor)) {
                    particle.resolveCollision(*neighbor);
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        cumulativeTime += end - start;
        spatialHashOps += num_ops;
    } else if (withTree) {

        // copy quadtree particles to array
        std::vector<Particle> treeParticles = quadtree.getParticles();
        std::copy(treeParticles.begin(), treeParticles.end(), particles);

        quadtree.clear();

        for (int i = 0; i < num_particles; i++) {
            // Render the particle
            particles[i].render();
            // Update the particle's position, check for wall collision
            particles[i].updatePosition(delta);
            particles[i].wallBounce();
            // Repopulate quadtree
            quadtree.insert(particles[i]);
        }

        // Check for and resolve collisions
        for (int i = 0; i < num_particles; i++) {
            quadtree.checkCollisions(particles[i]);
        }
    } else {
        int num_ops = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_particles; i++) {
            // Render the particle
            particles[i].render();
            // Update the particle's position, check for wall collision
            particles[i].updatePosition(delta);
            particles[i].wallBounce();
            // // Check for collisions with other particles
            for (int j = 0; j < num_particles; j++) {
                if (particles[i].collidesWith(particles[j])) {
                    particles[i].resolveCollision(particles[j]);
                }
                num_ops += 1;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        cumulativeTime += end - start;
        bruteForceOps += num_ops;
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

int main(int argc, char** argv) {

    // Set defaults
    srand(time(NULL));

    num_particles = DEFAULT_P_NUMBER;
    particle_size = DEFAULT_P_SIZE;
    int opt;
    bool explode = false;
    withSweep = false;
    withTree = false;
    withSpatialHash = false;
    print_fps = false;

    // Command line options
    while ((opt = getopt(argc, argv, "n:s:ewhtgp")) != -1) {
        switch (opt) {
            case 'n':
                num_particles = strtol(optarg, NULL, 10);
                break;
            case 's':
                particle_size = strtod(optarg, NULL);
                break;
            case 'e':
                // Explode particles from center. Recommend running with a lot of particles with a low size
                explode = true;
                break;
            case 'w':
                if (withTree || withSpatialHash) {
                    fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-h help (OPTIONAL)]\n", argv[0]);
                    exit(EXIT_FAILURE);
                }
                withSweep = true;
                break;
            case 't':
                if (withSweep || withSpatialHash) {
                    fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-t with_tree (OPTIONAL)] [-g with_spatial_hash (OPTIONAL)] [-h help (OPTIONAL)]\n", argv[0]);
                    exit(EXIT_FAILURE);
                }
                withTree = true;
                break;
            case 'g':
                if (withSweep || withTree) {
                    fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-t with_tree (OPTIONAL)] [-g with_spatial_hash (OPTIONAL)] [-h help (OPTIONAL)]\n", argv[0]);
                    exit(EXIT_FAILURE);
                }
                withSpatialHash = true;
                break;
            case 'p':
                print_fps = true;
            case 'h':
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-t with_tree (OPTIONAL)] [-g with_spatial_hash (OPTIONAL)] [-h help (OPTIONAL)]\n", argv[0]);
                exit(EXIT_FAILURE);
            default:
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-t with_tree (OPTIONAL)] [-g with_spatial_hash (OPTIONAL)] [-h help (OPTIONAL)]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    particles = (Particle*) calloc(num_particles, sizeof(Particle));
    num_edges = num_particles * 2;
    edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));
    p_overlaps = new std::unordered_set<int>[num_particles];

    if (withTree) {
        float x = 0.0f;
        float y = 0.0f;
        float width = 100.0f;
        float height = 100.0f;
        int level = 0;
        int maxLevel = 4;

        // initialize the quadtree
        quadtree = Quadtree(X_MIN, Y_MIN, X_MAX - X_MIN, Y_MAX - Y_MIN, level, maxLevel);
    }

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

        float x, y;
        if (explode) {
            x = (X_MAX + X_MIN) / 2;
            y = (Y_MAX + Y_MIN) / 2;
        } else {
            x = pos_x(gen);
            y = pos_y(gen);
        }

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), mass(gen), particle_size);

        // // Insert the particle into the quadtree if it is enabled
        // if (withTree) {
        //     quadtree.insert(particles[i]);
        //             printf("INSERT\n");

        // }
    }
    // Initialize the list of edges, then sort them to prime the list for near-O(n) sorts.
    for (int i = 0; i < num_particles; i++) {
        edgesByX[i*2] = Edge(i, false);
        edgesByX[i*2 + 1] = Edge(i, true);
    }
    // TEST - verify if this sort is necessary
    if (withSweep) {
        sortByX(edgesByX);
    } else if (withTree) {
        for (int i = 0; i < num_particles; i++) {
            quadtree.insert(particles[i]);
            printf("INSERT\n");
        }
    }


    initGL(&argc, argv);
    lastTime = 0;
    glutMainLoop();

    return EXIT_SUCCESS;
}