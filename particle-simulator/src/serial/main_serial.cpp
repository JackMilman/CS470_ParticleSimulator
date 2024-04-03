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

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle_serial.h"
#include "particle_serial.cpp"
#include "vector_serial.h"
#include "vector_serial.cpp"
#include "edge.cpp"
#include "edge.h"

#include <math.h>
#define PI 3.14159265f

int num_particles;
float particle_size;
Particle* particles;

Edge* edgesByX;
int num_edges;

int lastTime;

// GL functionality
bool initGL(int *argc, char **argv);

void sortByX(Edge* edges) {
    // Simple insertion sort for the particles, sorting by their x-positions. This is to be used in sweep-and-prune.
    for (int i = 1; i < num_edges; i++) {
        for (int j = i - 1; j >= 0; j--) {
            Particle p_j = particles[edges[j].getParentIdx()];
            Particle p_next_j = particles[edges[j + 1].getParentIdx()];

            float j_x = p_j.getPosition().getX();
            if (edges[j].getIsLeft()) j_x -= p_j.getRadius();
            else j_x += p_j.getRadius();

            float j_next_x = p_next_j.getPosition().getX();
            if (edges[j + 1].getIsLeft()) j_next_x -= p_next_j.getRadius();
            else j_next_x += p_next_j.getRadius();

            if (j_x < j_next_x) break;
            Edge tmp = edges[j];
            edges[j] = edges[j + 1];
            edges[j + 1] = tmp;
        }
    }
    // for (int i = 0; i < num_edges; i++) {
    //     printf("(X:%0.02f Center: %0.02f) ", particles[edges[i].getParentIdx()].getPosition().getX(), particles[edges[i].getParentIdx()].getPosition().getX());
    // }
    // printf("\n");
}



inline bool operator<(const Particle& lhs, const Particle& rhs)
{
  return lhs.getID() < rhs.getID();
}

void sweepAndPruneByX() {
    sortByX(edgesByX);
    std::set<int> touching ; // indexes of particles touched by the line at this point
    for (int i = 0; i < num_edges; i++) {
        if (edgesByX[i].getIsLeft()) {
            for (auto itr : touching) {
                // edgesByX[i].getParent().resolveCollision(itr);
                Particle& p_edge = particles[edgesByX[i].getParentIdx()];
                Particle& p_other = particles[itr];
                p_edge.resolveCollision(p_other);
            }
            touching.insert(i);
        } else {
            touching.erase(i);
        }
    }

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

// OpenGL rendering
void display() {
	glClear(GL_COLOR_BUFFER_BIT);
    
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


    for (int i = 0; i < num_particles; i++) {
        // Render the particle
        particles[i].renderCircle();
        // Update the particle's position, check for wall collision
        particles[i].updatePosition(delta);
        particles[i].wallBounce();

        // // Check for collisions with other particles
        // for (int j = i + 1; j < num_particles; j++) {
        //     if (particles[i].collidesWith(particles[j])) {
        //         particles[i].resolveCollision(particles[j]);
        //     }
        // }
        sweepAndPruneByX();
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
    num_particles = 10;
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
    num_edges = num_particles * 2;
    edgesByX = (Edge*) calloc(num_edges, sizeof(Edge));

    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Randomize velocity, position, and mass
        std::uniform_real_distribution<float> dist(-2, 2);
        std::uniform_real_distribution<float> rand(-0.95, 0.95);
        std::uniform_real_distribution<float> mass(1.5, 5);

        // make random particle velocity        
        float dx = dist(gen);
        float dy = dist(gen);

        float x, y;
        if (explode) {
            x = 0;
            y = 0;
        } else {
            x = rand(gen);
            y = rand(gen);
        }

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), mass(gen), particle_size, i);
    }

    for (int i = 0; i < num_particles; i++) {
        edgesByX[i*2] = Edge(i, false);
        edgesByX[i*2 + 1] = Edge(i, true);
    }
    sortByX(edgesByX);

    initGL(&argc, argv);
    lastTime = 0;
    glutMainLoop();

    return 0;
}