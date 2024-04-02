#include <cuda_runtime.h>
#include <GL/gl.h>     // The GL Header File
#include <GL/glut.h>   // The GL Utility Toolkit (Glut) Header
#include <math.h>

#include "particle.cuh"

#define PI 3.14159265f

Particle::Particle() : position(Vector(0, 0, 0)), velocity(Vector(0, 0, 0)), mass(1), radius(1) {}
Particle::Particle(const Vector& position, const Vector& velocity, float mass, float radius ) : position(position), velocity(velocity), mass(mass), radius(radius) {}

__host__ __device__ const Vector& Particle::getPosition() const {
    return position;
}

__device__ void Particle::setPosition(const Vector& position) {
    this->position = position;
    }

__device__ const Vector& Particle::getVelocity() const {
    return velocity;
}

__device__ void Particle::setVelocity(const Vector& new_velocity) {
    this->velocity = new_velocity;
}

__device__ float Particle::getMass() const {
    return mass;
}

__device__ void Particle::setMass(float mass) {
    this->mass = mass;
}

__device__ float Particle::getRadius() const {
    return radius;
}

__device__ void Particle::setRadius(float radius) {
    this->radius = radius;
}

__device__ void Particle::updatePosition(float deltaTime) {
    this->position += this->velocity * deltaTime;
}

__host__ void Particle::renderSphere() {
    GLfloat ballRadius = (GLfloat) this->radius;   // Radius of the bouncing ball
    GLfloat ballX = (GLfloat) this->position.getX();
    GLfloat ballY = (GLfloat) this->position.getY();
    GLfloat ballZ = (GLfloat) this->position.getZ();

    glMatrixMode(GL_MODELVIEW);    // To operate on the model-view matrix
    glLoadIdentity();              // Reset model-view matrix

    glTranslatef(ballX, ballY, ballZ);  // Translate to (xPos, yPos)

    glColor3f(1, 0, 1);

    glutSolidSphere(ballRadius, 1, 1);
    // Use triangular segments to form a circle
    // glBegin(GL_TRIANGLE_FAN);
    //     glColor3f(1, 0, 1);
    //     glVertex2f(0.0f, 0.0f);       // Center of circle
    //     int numSegments = 100;
    //     GLfloat angle;
    //     for (int i = 0; i <= numSegments; i++) { // Last vertex same as first vertex
    //         angle = (i * 2.0f * PI) / numSegments;  // 360 deg for all segments
    //         glVertex2f(cos(angle) * ballRadius, sin(angle) * ballRadius);
    //     }
    // glEnd();
}

__device__ void Particle::wallBounce() {
    float x = this->position.getX();
    float y = this->position.getY();
    float z = this->position.getZ();
    float dx = this->velocity.getX();
    float dy = this->velocity.getY();
    float dz = this->velocity.getZ();
    float radius = this->getRadius();
    
    if (x + radius > 1) {
        this->position.setX(1 - radius);
        this->velocity.setX(-dx);
    } else if (x - radius < -1) {
        this->position.setX(-1 + radius);
        this->velocity.setX(-dx);
    }

    if (y + radius > 1) {
        this->position.setY(1 - radius);
        this->velocity.setY(-dy);
    } else if (y - radius < -1) {
        this->position.setY(-1 + radius);
        this->velocity.setY(-dy);
    }

    if (z + radius > 1) {
        this->position.setZ(1 - radius);
        this->velocity.setZ(-dz);
    } else if (z - radius < -1) {
        this->position.setZ(-1 + radius);
        this->velocity.setZ(-dz);
    }
}

__device__ bool Particle::collidesWith(const Particle& other) const {
    Vector p1Pos = this->getPosition();
    Vector p2Pos = other.getPosition();
    float p1Radius = this->getRadius();
    float p2Radius = other.getRadius();
    float dx = p1Pos.getX() - p2Pos.getX();
    float dy = p1Pos.getY() - p2Pos.getY();
    float dz = p1Pos.getZ() - p2Pos.getZ();
    float squaredDistance = dx * dx + dy * dy + dz * dz;

    float radiiSum = p1Radius + p2Radius;
    float squaredSumOfRadii = radiiSum * radiiSum;
    bool collision = squaredDistance < squaredSumOfRadii;
    return collision;
}

__device__ void Particle::resolveCollision(Particle& other) {
    // With help from https://stackoverflow.com/questions/345838/ball-to-ball-collision-detection-and-handling
    Vector p1Pos = this->getPosition();
    Vector p2Pos = other.getPosition();

    float distance = sqrt(pow(p1Pos.getX() - p2Pos.getX(), 2) + pow(p1Pos.getY() - p2Pos.getY(), 2) + pow(p1Pos.getZ() - p2Pos.getZ(), 2));
    Vector collision;
    if (distance == 0) {
        collision = Vector(1, 0, 0);  // Avoid division by zero; arbitrary collision vector
        distance = 1;
    } else {
        collision = (p1Pos - p2Pos) / distance;  // Normalized collision vector
    }

    // components of velocity along collision vector
    double aci = this->getVelocity().dot(collision);
    double bci = other.getVelocity().dot(collision);

    // Set final velocities
    double acf = (aci * (this->getMass() - other.getMass()) + 2 * other.getMass() * bci) / (this->getMass() + other.getMass());
    double bcf = (bci * (other.getMass() - this->getMass()) + 2 * this->getMass() * aci) / (this->getMass() + other.getMass());

    this->setVelocity((this->getVelocity() + collision * (acf - aci) * 1 / this->getMass()));
    other.setVelocity((other.getVelocity() + collision * (bcf - bci) * 1 / other.getMass()));

    // Prevent particles from overlapping
    float radiiSum = this->getRadius() + other.getRadius();
    float overlap = radiiSum - distance;
    if (overlap > 0) {
        float overlap1 = overlap * other.getMass() / (this->getMass() + other.getMass());
        float overlap2 = overlap * this->getMass() / (this->getMass() + other.getMass());
        this->setPosition(this->getPosition() + collision * overlap1);
        other.setPosition(other.getPosition() - collision * overlap2);
    }
    
}

