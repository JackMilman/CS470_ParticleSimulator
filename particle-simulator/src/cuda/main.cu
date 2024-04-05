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

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle.cuh"
#include "particle.cu"
#include "vector.cuh"
#include "vector.cu"
#include "edge.cu"
#include "edge.cuh"

#include <curand.h>
#include <curand_kernel.h>

int num_particles;
float particle_size;
Particle* particles;
Particle* device_particles;
curandState* states;

Edge* edgesByX;
int num_edges;
bool withSweep;
std::unordered_set<int>* p_overlaps;
std::unordered_set<int>* device_overlaps;

int lastTime;

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

// Sweeps across the list of particle edges, sorted by their minimum x-values. If an edge is a left-edge, 
// we look at all the other particles currently being "touched" by our imaginary line and check if they
// have already been resolved. If they have not yet been resolved, we perform a finer-grained check to 
// see if they collide, and resolve a collision if they do.
void sweepAndPruneByX() {
    sortByX(edgesByX);
    std::unordered_set<int> touching; // indexes of particles touched by the line at this point
    int p_edge_idx;
    int checked = 0;
    for (int i = 0; i < num_edges; i++) {
        p_edge_idx = edgesByX[i].getParentIdx();
        if (edgesByX[i].getIsLeft()) {
            for (auto itr = touching.begin(); itr != touching.end(); ++itr) {
                bool checked = resolved(p_edge_idx, *itr);
                if (!checked) {
                    // if (particles[p_edge_idx].collidesWith(particles[*itr])) {
                    //     particles[p_edge_idx].resolveCollision(particles[*itr]);                      
                    // }
                    p_overlaps[p_edge_idx].insert(*itr);
                    p_overlaps[*itr].insert(p_edge_idx);
                    checked += 1;
                }
            }
            touching.insert(p_edge_idx);
        } else {
            touching.erase(p_edge_idx);
        }
    }
    // // Resets the overlapping pairs sets for the next iteration of the algorithm.
    // for (int i = 0; i < num_particles; i++) {
    //     p_overlaps[i].clear();
    // }
    // printf("Particles: %d\n", num_particles);
    // printf("Checked: %d\n", checked);
}

__global__ void checkSweep(Particle* d_particles, std::unordered_set<int>* d_overlaps, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto itr = d_overlaps[i].begin(); itr != d_overlaps[i].end(); ++itr) {
        if (d_particles[i].collidesWith(d_particles[*itr])) {
            d_particles[i].resolveCollision(d_particles[*itr]);
        }
    }
}

// Check for collisions and resolve them
__global__ void checkCollision(Particle* d_particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < n_particles; j++) {
        if (d_particles[i].collidesWith(d_particles[j])) {
            d_particles[i].resolveCollision(d_particles[j]);
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

    // Render particles
    for (int i = 0; i < num_particles; i++) {
        particles[i].renderSphere();
    }

    int blockSize = 256;
    int blockCount = (num_particles + blockSize - 1) / blockSize;

    // FPS counter
    static int frameCount = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    float delta = (currentTime - lastTime) / 1000.0f;
    lastTime = currentTime;
    frameCount++;

    if (frameCount % 20 == 0) {
        char title[80];
        sprintf(title, "Particle Simulator (%.2f fps) - %d particles", 1 / delta, num_particles);
        printf("%f\n", 1 / delta);
        glutSetWindowTitle(title);
    }

    // Send particle data to device
    cudaMemcpy(device_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
    updateParticles<<<blockCount, blockSize>>>(device_particles, num_particles, delta);
    cudaDeviceSynchronize();
    if (withSweep) {
        sweepAndPruneByX();
        cudaMemcpy(device_overlaps, p_overlaps, sizeof(p_overlaps), cudaMemcpyHostToDevice);
        checkSweep<<<blockCount, blockSize>>>(device_particles, device_overlaps, num_particles);
        cudaMemcpy(p_overlaps, device_overlaps, sizeof(device_overlaps), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_particles; i++) {
            p_overlaps[i].clear();
        }
    } else {
        checkCollision<<<blockCount, blockSize>>>(device_particles, num_particles);
    }
    
    // Retrieve particle data from device
    cudaMemcpy(particles, device_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // sweepAndPrune();

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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Setup perspective projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 1.0, 0.1, 10.0);

    // Setup the camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(100.0, 0.0, 100.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);

    return true;
}

int main(int argc, char** argv) {
    // Set defaults
    srand(time(NULL));
    num_particles = 10;
    particle_size = 0.01f;
    int opt;
    bool explode = false;

    // Command line options
    while ((opt = getopt(argc, argv, "n:s:ewh")) != -1) {
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
                withSweep = true;
                break;
            case 'h':
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)]\n", argv[0]);
                exit(EXIT_FAILURE);
            default:
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    particles = (Particle*) calloc(num_particles, sizeof(Particle));
    num_edges = num_particles * 2;
    edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));
    p_overlaps = new std::unordered_set<int>[num_particles];

    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Randomize velocity, position, depth, and mass
        std::uniform_real_distribution<float> velocity(-2, 2);
        std::uniform_real_distribution<float> position_x(X_MIN + particle_size, X_MAX - particle_size);
        std::uniform_real_distribution<float> position_y(Y_MIN + particle_size, Y_MAX - particle_size);
        std::uniform_real_distribution<float> position_z(Z_MIN + particle_size, Z_MAX - particle_size);
        std::uniform_real_distribution<float> mass(1.5, 5);

        float dx = velocity(gen);
        float dy = velocity(gen);
        float dz = velocity(gen);  // z-velocity

        float x, y, z;
        if (explode) {
            x = (X_MAX + X_MIN) / 2;
            y = (Y_MAX + Y_MIN) / 2;
            z = (Z_MAX + Z_MIN) / 2;  // Explode from center
        } else {
            x = position_x(gen);
            y = position_y(gen);
            z = position_z(gen);  // z-coordinate
        }

        particles[i] = Particle(Vector(x, y, z), Vector(dx, dy, dz), mass(gen), particle_size);
    }
    for (int i = 0; i < num_particles; i++) {
        edgesByX[i*2] = Edge(i, false);
        edgesByX[i*2 + 1] = Edge(i, true);
    }

    // Init the device particles
    cudaMalloc((void**)&device_particles, num_particles * sizeof(Particle));
    cudaMalloc((void**)&device_overlaps, num_particles * sizeof(p_overlaps));
    // cudaMalloc((void**)&states, num_particles * sizeof(curandState));

    initGL(&argc, argv);

    lastTime = 0;
    glutMainLoop();

    // Clean up
    cudaDeviceSynchronize();
    cudaFree(device_particles);
    cudaFree(device_overlaps);
    // cudaFree(states);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
struct OctreeNode {
    int children[8]; // Indices of the children nodes in the nodes array. -1 if no child.
    float3 min; // Minimum point of the bounding box
    float3 max; // Maximum point of the bounding box
    int particleStart; // Index of the first particle in this node in the particles array
    int particleEnd; // Index of the last particle in this node in the particles array
};

struct Octree {
    OctreeNode* nodes; // Array of nodes
    Particle* particles; // Array of particles
    int numNodes; // Number of nodes in the tree
    int numParticles; // Number of particles in the tree
};

void buildOctree(Octree* octree, Particle* particles, int numParticles) {
    // Allocate memory for the nodes and particles arrays
    octree->nodes = new OctreeNode[numParticles * 8]; // This is an overestimate, but it ensures we won't run out of space
    octree->particles = new Particle[numParticles];

    // Copy the particles into the Octree
    memcpy(octree->particles, particles, numParticles * sizeof(Particle));
    octree->numParticles = numParticles;

    // Create the root node
    octree->numNodes = 1;
    octree->nodes[0].min = make_float3(X_MIN, Y_MIN, Z_MIN);
    octree->nodes[0].max = make_float3(X_MAX, Y_MAX, Z_MAX);
    octree->nodes[0].particleStart = 0;
    octree->nodes[0].particleEnd = numParticles;
}
/////////////////////////////////////////////////////////////////////////////////////

///////////////////////     OctTree Functions (Need more testing)       //////////////////////////////
// __global__ void queryOctree(Particle* particles, OctreeNode* nodes, int* queue, bool* visited, int numParticles, int numNodes, float interactionRadius) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;

//     if (index < numParticles) {
//         Particle* particle = &particles[index];

//         // Initialize the queue with the root node
//         if (index == 0) {
//             queue[0] = 0; // Assume the root node is at index 0
//         }

//         __syncthreads();

//         // BFS traversal of the Octree
//         for (int i = 0; i < numNodes; i++) {
//             if (i < numNodes && !visited[i]) {
//                 OctreeNode* node = &nodes[queue[i]];

//                 // Check particles at this octree level
//                 for (int j = node->particleStart; j < node->particleEnd; j++) {
//                     Particle* p = &particles[j];
//                     if (distance(particle->position, p->position) <= interactionRadius) {
//                         // Handle interaction between particle and p
//                     }
//                 }

//                 // If this node has children, add them to the queue
//                 for (int j = 0; j < 8; j++) {
//                     int childIndex = node->children[j];
//                     if (childIndex != -1 && intersectsSphere(node->min, node->max, particle->position, interactionRadius)) {
//                         queue[numNodes++] = childIndex;
//                     }
//                 }

//                 visited[i] = true;
//             }

//             __syncthreads();
//         }
//     }
// }

// __global__ void updateParticlePositions(Particle* particles, int numParticles, float dt) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;

//     if (index < numParticles) {
//         Particle* particle = &particles[index];

//         // Update the particle's position based on its velocity and the time step
//         particle->position.x += particle->velocity.x * dt;
//         particle->position.y += particle->velocity.y * dt;
//         particle->position.z += particle->velocity.z * dt;
//     }
// }

// void updateOctree(Particle* particles, OctreeNode* nodes, int numParticles, int numNodes) {
//     // Rebuild the Octree on the CPU. This could be parallelized on the GPU, but it's a complex task that's beyond the scope of this example.
//     // ...
// }

// void simulate(Particle* particles, OctreeNode* nodes, int numParticles, int numNodes, float dt) {
//     // Update the particle positions
//     int blockSize = 256;
//     int numBlocks = (numParticles + blockSize - 1) / blockSize;
//     updateParticlePositions<<<numBlocks, blockSize>>>(particles, numParticles, dt);
//     cudaDeviceSynchronize();

//     // Rebuild the Octree
//     updateOctree(particles, nodes, numParticles, numNodes);
// }