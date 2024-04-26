#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

#include "vector.cuh"

#define BALL_SEGMENTS 50
#define PI 3.14159265f
#define VEL_MIN -1.0
#define VEL_MAX 1.0
#define X_MAX 1.0
#define X_MIN -1.0
#define Y_MAX 1.0
#define Y_MIN -1.0
// #define Z_MAX -2.0
// #define Z_MIN -6.0

class Particle {
public:
    Particle();
    Particle(const Vector& position, const Vector& velocity, float mass, float radius);

    // Getters and Setters
    __host__ __device__ const Vector& getPosition() const;
    __device__ void setPosition(const Vector& position);

    __device__ const Vector& getVelocity() const;
    __device__ void setVelocity(const Vector& velocity);

    __device__ float getMass() const;
    __device__ void setMass(float mass);

    __host__ __device__ float getRadius() const;
    __device__ void setRadius(float radius);

    // Other methods
    __device__ void updatePosition(float dt);
    __host__ void render();
    __device__ void wallBounce();

    __device__ bool collidesWith(const Particle& other) const;
    __device__ void resolveCollision(Particle& other);
    //void updateVelocity(const Vector& force, float deltaTime);
private:
    Vector position;
    Vector velocity;
    float mass;
    float radius;
};



#endif // PARTICLE_H