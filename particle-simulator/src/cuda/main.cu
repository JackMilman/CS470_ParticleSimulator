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

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle.cuh"
#include "particle.cu"
#include "vector.cuh"
#include "vector.cu"

#include <curand.h>
#include <curand_kernel.h>

int num_particles;
float particle_size;
Particle* particles;
Particle* device_particles;
curandState* states;

int lastTime;

// GL functionality
bool initGL(int *argc, char **argv);

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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
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
    gluLookAt(1000.0, 0.0, 1000.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);

    return true;
    // glutInit(argc, argv);
    // glutInitWindowSize(800, 800);
    // glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    // glutCreateWindow("Particle Simulator");
    // glutPositionWindow(950,100);
    // glutTimerFunc( 0, timer, 0 );
    // glutDisplayFunc(display);

    // // Initialize GLEW
    // glewExperimental = GL_TRUE;
    // GLenum err = glewInit();
    // if (err != GLEW_OK) {
    //     fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
    //     return false;
    // }

    // return true;
}

int main(int argc, char** argv) {
    // Set defaults
    srand(time(NULL));
    num_particles = 100;
    particle_size = 0.1f;
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

    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Randomize velocity and position in 3D
        std::uniform_real_distribution<float> dist(-2, 2);
        std::uniform_real_distribution<float> randPosition(-0.95, 0.95);
        std::uniform_real_distribution<float> mass(1.5, 5);

        float dx = dist(gen);
        float dy = dist(gen);
        float dz = dist(gen);  // z-velocity

        float x, y, z;
        if (explode) {
            x = y = z = 0;  // Explode from center
        } else {
            x = randPosition(gen);
            y = randPosition(gen);
            z = randPosition(gen);  // z-coordinate
        }

        particles[i] = Particle(Vector(x, y, z), Vector(dx, dy, dz), mass(gen), particle_size);
    }


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