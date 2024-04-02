#include "vector.cuh"

// Default constructor
__host__ __device__ Vector::Vector() : x(0.0f), y(0.0f), z(0.0f) {}

__host__ __device__ Vector::Vector(float x, float y, float z) : x(x), y(y), z(z) {}

__host__ __device__ float Vector::getX() const {
    return x;
}

__host__ __device__ void Vector::setX(float x) {
    this->x = x;
}

__host__ __device__ float Vector::getY() const {
    return y;
}

__host__ __device__ void Vector::setY(float y) {
    this->y = y;
}

__host__ __device__ float Vector::getZ() const {
    return z;
}

__host__ __device__ void Vector::setZ(float z) {
    this->z = z;
}

__host__ __device__ float Vector::dot(const Vector& other) const {
    return x * other.x + y * other.y + z * other.z;
}

__host__ __device__ Vector Vector::operator+(const Vector& other) const {
    return Vector(x + other.x, y + other.y, z + other.z);
}

__host__ __device__ Vector Vector::operator+=(const Vector& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

__host__ __device__ Vector Vector::operator-(const Vector& other) const {
    return Vector(x - other.x, y - other.y, z - other.z);
}

__host__ __device__ Vector Vector::operator-=(const Vector& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

__host__ __device__ Vector Vector::operator*(float scalar) const {
    return Vector(x * scalar, y * scalar, z * scalar);
}

__host__ __device__ Vector& Vector::operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

__host__ __device__ Vector Vector::operator/(float scalar) const {
    return Vector(x / scalar, y / scalar, z / scalar);
}

__host__ __device__ Vector& Vector::operator/=(float scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
}
