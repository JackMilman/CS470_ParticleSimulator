#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <stack>
#include <unistd.h>
#include <set>
#include <unordered_set>
#include <chrono>
#include <iomanip>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle.cuh"
#include "particle.cu"
#include "vector.cuh"
#include "vector.cu"
#include "edge.cu"
#include "edge.cuh"
#include "particle_pair.cu"
#include "particle_pair.cuh"
#include "quadtree.cu"
// #include "quadtree.cuh"
#include "spatial_hashing.cu"
#include "spatial_hashing.cuh"

#include <curand.h>
#include <curand_kernel.h>

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
Particle* device_particles;

enum modes {BruteForce, SweepAndPrune, Quad, Hash};
int mode = BruteForce;

// Rectangle rectangle = Rectangle((float) X_MIN, (float) Y_MIN, (float) X_MAX, (float) Y_MAX);
// QuadTree quadtree = QuadTree(0, rectangle);
Edge* edgesByX;
int num_edges;
int max_pairs;
std::unordered_set<int>* p_overlaps;
std::unordered_set<int>* device_overlaps;
ParticlePair* pairs;
ParticlePair* device_pairs;

float cellSize = DEFAULT_P_SIZE;
SpatialHash spatialHash(cellSize);

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
int sweepAndPruneByX() {
    sortByX(edgesByX);
    // indexes of particles touched by the line at this point
    std::unordered_set<int> touching;
    int p_edge_idx;
    // pair_idx represents the number of overlaps (minus one) that we have found in
    // this iteration. It will be used to determine how far into the pairs array to
    // look, when fine-checking for collisions
    int pair_idx = 0;
    for (int i = 0; i < num_edges; i++) {
        p_edge_idx = edgesByX[i].getParentIdx();
        if (edgesByX[i].getIsLeft()) {
            for (auto itr = touching.begin(); itr != touching.end(); ++itr) {
                    pairs[pair_idx] = ParticlePair(p_edge_idx, *itr);
                    pair_idx++;
            }
            touching.insert(p_edge_idx);
        } else {
            touching.erase(p_edge_idx);
        }
    }
    return pair_idx - 1;
}


// Check for collisions and resolve them
__global__ void checkBruteForce(Particle* d_particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = i + 1; j < n_particles; j++) {
        if ((i != j) && d_particles[i].collidesWith(d_particles[j])) {
            d_particles[i].resolveCollision(d_particles[j]);
        }
    }
}

__global__ void checkSweep(Particle* d_particles, ParticlePair* d_pairs, int n_pairs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n_pairs; i+=stride) {
        Particle& a = d_particles[d_pairs[i].getA()];
        Particle& b = d_particles[d_pairs[i].getB()];
        if (a.collidesWith(b)) {
            a.resolveCollision(b);
        }
    }
}

// Update the position of the particles and check for wall collisions
__global__ void updateParticles(Particle* d_particles, int n_particles, float deltaTime) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        d_particles[i].updatePosition(deltaTime);
        d_particles[i].wallBounce();
    }
}

// Host function
void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // FPS counter
    static int frameCount = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    float delta = (currentTime - lastTime) / 1000.0f;
    lastTime = currentTime;
    frameCount++;

    // Render particles
    for (int i = 0; i < num_particles; i++) {
        particles[i].render();
    }

    int blockSize = 256;
    int blockCount = (num_particles + blockSize - 1) / blockSize;

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
    int n_pairs = 0;

    int storageSize = 0;
    int* d_storageSize;
    cudaMalloc((void**)&d_storageSize, sizeof(int));
    cudaMemcpy(d_storageSize, &storageSize, sizeof(int), cudaMemcpyHostToDevice);

    int* d_keys, *d_particleIndices;
    cudaMalloc((void**)&d_keys, num_particles * sizeof(int));
    cudaMalloc((void**)&d_particleIndices, num_particles * sizeof(int));

    // Send particle data to device
    cudaMemcpy(device_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
    updateParticles<<<blockCount, blockSize>>>(device_particles, num_particles, delta);
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    switch (mode) {
        case BruteForce:
            checkBruteForce<<<blockCount, blockSize>>>(device_particles, num_particles);
            break;
        case SweepAndPrune:
            n_pairs = sweepAndPruneByX();
            cudaMemcpy(device_pairs, pairs, max_pairs * sizeof(ParticlePair), cudaMemcpyHostToDevice);
            checkSweep<<<blockCount, blockSize>>>(device_particles, device_pairs, n_pairs);
            cudaMemcpy(pairs, device_pairs, max_pairs * sizeof(ParticlePair), cudaMemcpyDeviceToHost);
            break;
        case Quad:
            ////////////    QuadTree Implementation - See analysis for explanation on CPU-GPU dependencies      ////////////
            break;
        case Hash:
            spatialHash.clear();
            insertParticles<<<blockCount, blockSize>>>(device_particles, num_particles, cellSize, d_storageSize, d_keys, d_particleIndices);
            cudaDeviceSynchronize();
            queryParticles<<<blockCount, blockSize>>>(device_particles, num_particles, cellSize, d_storageSize, d_keys, d_particleIndices);
            break;
    }
    cudaMemcpy(&storageSize, d_storageSize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Retrieve particle data from device
    cudaMemcpy(particles, device_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
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

    cudaFree(d_keys);
    cudaFree(d_particleIndices);
    cudaFree(d_storageSize);

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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("3D Particle Simulator");
    glutPositionWindow(950,100);
    glutTimerFunc( 0, timer, 0 );
    glutDisplayFunc(display);

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
                // rectangle = new Rectangle((float) X_MIN, (float) Y_MIN, (float) X_MAX, (float) Y_MAX);
                // quadtree = new QuadTree(0, *rectangle);
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
    // num_edges = num_particles * 2;
    // edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));
    // p_overlaps = new std::unordered_set<int>[num_particles];
    // max_pairs = num_particles * num_particles;
    // pairs = (ParticlePair*) calloc(max_pairs, sizeof(ParticlePair));


    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Randomize velocity, position, depth, and mass
        std::uniform_real_distribution<float> velocity(VEL_MIN, VEL_MAX);
        std::uniform_real_distribution<float> position_x(X_MIN + particle_size, X_MAX - particle_size);
        std::uniform_real_distribution<float> position_y(Y_MIN + particle_size, Y_MAX - particle_size);
        std::uniform_real_distribution<float> mass(1.5, 5);

        float dx = velocity(gen);
        float dy = velocity(gen);

        float x, y;
        if (explode) {
            x = (X_MAX + X_MIN) / 2;
            y = (Y_MAX + Y_MIN) / 2;
        } else {
            x = position_x(gen);
            y = position_y(gen);
        }

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), mass(gen), particle_size);
    }
    
    // Init the device particles
    cudaMalloc((void**)&device_particles, num_particles * sizeof(Particle));
    switch (mode) {
        case SweepAndPrune:
            for (int i = 0; i < num_particles; i++) {
                edgesByX[i*2] = Edge(i, false);
                edgesByX[i*2 + 1] = Edge(i, true);
            }
            sortByX(edgesByX);
            cudaMalloc((void**)&device_overlaps, num_particles * sizeof(p_overlaps));
            cudaMalloc((void**)&device_pairs, max_pairs * sizeof(ParticlePair));
            break;
        case Hash:
            SpatialHash* spatialHash;
            cudaMalloc(&spatialHash, sizeof(SpatialHash));
            float cellSize = 2 * particle_size;  // Cell size can be twice the particle size
            SpatialHash newHash(cellSize);
            cudaMemcpy(spatialHash, &newHash, sizeof(SpatialHash), cudaMemcpyHostToDevice);
            break;
    }
    

    

    
    initGL(&argc, argv);

    lastTime = 0;
    glutMainLoop();

    // Clean up
    cudaDeviceSynchronize();
    cudaFree(device_particles);
    cudaFree(device_pairs);

    return 0;
}