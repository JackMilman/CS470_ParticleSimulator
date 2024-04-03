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
#include <map>

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

int lastTime;

// GL functionality
bool initGL(int *argc, char **argv);

void sortByX(Edge* edges) {
    // Simple insertion sort for the particles, sorting by their x-positions. This is to be used in sweep-and-prune.
    for (int i = 1; i < num_edges; i++) {
        for (int j = i - 1; j >= 0; j--) {
            float j_x = edges[j].getX();
            float j_next_x = edges[j + 1].getX();
            if (j_x < j_next_x) break;
            Edge tmp = edges[j];
            edges[j] = edges[j + 1];
            edges[j + 1] = tmp;
        }
    }
    for (int i = 0; i < num_edges; i++) {
        printf("(X:%0.02f Parent-Center: %0.02f)", edges[i].getX(), edges[i].getParent().getPosition().getX());
    }
    printf("\n");
}

void sweepAndPruneByX() {

    // std::map<Edge, Particle> overlapping;
    // Edge* edges = edgesByX;
    // for (int i = 0; i < num_edges; i++) {
    //     for (int j = i - 1; j >= 0; j--) {
    //         float j_x = edges[j].getX();
    //         float j_next_x = edges[j + 1].getX();
    //         if (j_x < j_next_x) break;
    //         Edge tmp = edges[j];
    //         edges[j] = edges[j+1];
    //         edges[j+1] = tmp;

    //         Edge edge1 = edges[j];
    //         Edge edge2 = edges[j + 1];

    //         if (edge1.getIsLeft() && !edge2.getIsLeft()) {
    //             overlapping.insert(edge1, )
    //         } else if (!edge1.getIsLeft() && edge2.getIsLeft()) {
                
    //         }
    //     }
    // }

}

// Check for collisions and resolve them
__global__ void checkCollision(Particle* d_particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = i + 1; j < n_particles; j++) {
        if (d_particles[i].collidesWith(d_particles[j])) {
            d_particles[i].resolveCollision(d_particles[j]);
        }
    }
}

// Update the position of the particles and check for wall collisions
__global__ void updateParticles(Particle* d_particles, int n_particles, curandState* states, float deltaTime) {
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
    updateParticles<<<blockCount, blockSize>>>(device_particles, num_particles, states, delta);
    checkCollision<<<blockCount, blockSize>>>(device_particles, num_particles);
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
    while ((opt = getopt(argc, argv, "n:s:e")) != -1) {
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
            default:
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    particles = (Particle*) calloc(num_particles, sizeof(Particle));
    num_edges = num_particles * 2;
    edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));

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
        edgesByX[i*2] = Edge(particles[i], false);
        edgesByX[i*2 + 1] = Edge(particles[i], true);
    }
    sortByX(edgesByX);

    // Init the device particles
    cudaMalloc((void**)&device_particles, num_particles * sizeof(Particle));
    cudaMalloc((void**)&states, num_particles * sizeof(curandState));

    initGL(&argc, argv);

    lastTime = 0;
    glutMainLoop();

    // Clean up
    cudaDeviceSynchronize();
    cudaFree(device_particles);
    cudaFree(states);

    return 0;
}