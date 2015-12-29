#ifndef GEOM_H
#define GEOM_H

#include <math.h>

// amount squared to figure if two floats are equal
#define DELTA 1e-6

#define DEGRAD (180.0/M_PI) // degrees per radian
#define RADDEG (M_PI/180.0) // radians per degree


class Vec2f
{
 public:
    float p[2];
    Vec2f() { p[0]=0.0; p[1]=0.0; }
    Vec2f(float x, float y) { p[0]=x; p[1]=y; }
    Vec2f(const float *f) { p[0]=f[0]; p[1]=f[1]; }
    Vec2f(double *d) { p[0]=d[0]; p[1]=d[1]; }
    Vec2f(const Vec2f& v) { p[0]=v.p[0]; p[1]=v.p[1]; }
    inline void setValue(float x, float y) { p[0]=x; p[1]=y; }
    inline float* getValue() { return p; }
    inline void set(float *f) { p[0]=f[0]; p[1]=f[1]; }
    inline void scale(float s) { p[0]*=s; p[1]*=s; }
    inline double length() { return sqrt(p[0]*p[0]+p[1]*p[1]); }
    inline void normalize() { double l = length(); if (l) scale(1.0/l); }
    inline Vec2f normalized()
    { double l = length(); return Vec2f(p[0]/l,p[1]/l); }
    inline void setLength(double l) { scale(l/length()); }
    inline void negate() { p[0]=-p[0]; p[1]=-p[1]; }
    Vec2f perp() { return Vec2f(-p[1],p[0]); }
    float& operator[](int n)
    { return p[n]; }
    const float& operator[](int n) const
    { return p[n]; }
    Vec2f operator+(const Vec2f &v)
    { return Vec2f(p[0]+v.p[0], p[1]+v.p[1]); }
    Vec2f operator-(const Vec2f &v)
    { return Vec2f(p[0]-v.p[0], p[1]-v.p[1]); }
    Vec2f operator-()
    { return Vec2f(-p[0], -p[1]); }
    Vec2f operator*(double f)
    { return Vec2f(p[0]*f, p[1]*f); }
    Vec2f operator/(double f)
    { return Vec2f(p[0]/f, p[1]/f); }
    Vec2f& operator+=(const Vec2f& v)
    { p[0]+=v.p[0]; p[1]+=v.p[1]; return *this; }
    Vec2f& operator*=(double f)
    { p[0]*=f; p[1]*=f; return *this; }
    Vec2f& operator/=(double f)
    { p[0]/=f; p[1]/=f; return *this; }
    bool operator==(const Vec2f &v)
    { return fabs(p[0]-v.p[0])<DELTA && fabs(p[1]-v.p[1])<DELTA; }
    bool equals(const Vec2f &v, float tol) const
    { return fabs(p[0]-v.p[0])<tol && fabs(p[1]-v.p[1])<tol; }
    double dot(const Vec2f &v) const
    { return p[0]*v.p[0] + p[1]*v.p[1]; }
};


class Vec3f
{
 public:
    float p[3];
    Vec3f() { p[0]=0.0; p[1]=0.0; p[2]=0.0; }
    Vec3f(float x, float y, float z) { p[0]=x; p[1]=y; p[2]=z; }
    Vec3f(const float *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; }
    Vec3f(double *d) { p[0]=d[0]; p[1]=d[1]; p[2]=d[2]; }
    Vec3f(const Vec3f& v) { p[0]=v.p[0]; p[1]=v.p[1]; p[2]=v.p[2];}
    inline void setValue(float x, float y, float z) { p[0]=x; p[1]=y; p[2]=z; }
    inline float* getValue() { return p; }
    inline void set(float *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; }
    inline void scale(float s) { p[0]*=s; p[1]*=s; p[2]*=s; }
    inline double length() { return sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]); }
    inline void normalize() { double l = length(); if (l) scale(1.0/l); }
    inline Vec3f normalized()
    { double l = length(); return Vec3f(p[0]/l,p[1]/l,p[2]/l); }
    inline void setLength(double l) { scale(l/length()); }
    inline void negate() { p[0]=-p[0]; p[1]=-p[1]; p[2]=-p[2]; }
    float& operator[](int n)
    { return p[n]; }
    const float& operator[](int n) const
    { return p[n]; }
    Vec3f operator+(const Vec3f &v)
    { return Vec3f(p[0]+v.p[0], p[1]+v.p[1], p[2]+v.p[2]); }
    Vec3f operator-(const Vec3f &v)
    { return Vec3f(p[0]-v.p[0], p[1]-v.p[1], p[2]-v.p[2]); }
    Vec3f operator-()
    { return Vec3f(-p[0], -p[1], -p[2]); }
    Vec3f operator*(double f)
    { return Vec3f(p[0]*f, p[1]*f, p[2]*f); }
    Vec3f operator/(double f)
    { return Vec3f(p[0]/f, p[1]/f, p[2]/f); }
    Vec3f& operator+=(const Vec3f& v)
    { p[0]+=v.p[0]; p[1]+=v.p[1]; p[2]+=v.p[2]; return *this; }
    Vec3f& operator*=(double f)
    { p[0]*=f; p[1]*=f; p[2]*=f; return *this; }
    Vec3f& operator/=(double f)
    { p[0]/=f; p[1]/=f; p[2]/=f; return *this; }
    bool operator==(const Vec3f &v)
    { return fabs(p[0]-v.p[0])<DELTA
            && fabs(p[1]-v.p[1])<DELTA
            && fabs(p[2]-v.p[2])<DELTA; }
    bool equals(const Vec3f &v, float tol) const
    { return fabs(p[0]-v.p[0])<tol
            && fabs(p[1]-v.p[1])<tol
            && fabs(p[2]-v.p[2])<tol; }
    double dot(const Vec3f &v) const
    { return p[0]*v.p[0] + p[1]*v.p[1] + p[2]*v.p[2]; }
    Vec3f cross(const Vec3f &v) const
    { return Vec3f(p[1]*v.p[2] - p[2]*v.p[1],
		   p[2]*v.p[0] - p[0]*v.p[2],
		   p[0]*v.p[1] - p[1]*v.p[0]); }
    void rotateXY(float theta, float tx, float ty);
};

class Vec3i
{
 public:
    unsigned int p[3];
    Vec3i() { p[0]=0; p[1]=0; p[2]=0; }
    Vec3i(unsigned int x, unsigned int y, unsigned int z)
    { p[0]=x; p[1]=y; p[2]=z; }
    Vec3i(const unsigned int *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; }
    Vec3i(const Vec3i& v) { p[0]=v.p[0]; p[1]=v.p[1]; p[2]=v.p[2];}
    inline void setValue(unsigned int x, unsigned int y, unsigned int z) { p[0]=x; p[1]=y; p[2]=z; }
    inline unsigned int* getValue() { return p; }
    inline void set(unsigned int *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; }
    inline void scale(unsigned int s) { p[0]*=s; p[1]*=s; p[2]*=s; }
    inline void negate() { p[0]=-p[0]; p[1]=-p[1]; p[2]=-p[2]; }
    unsigned int& operator[](int n)
    { return p[n]; }
    const unsigned int& operator[](int n) const
    { return p[n]; }
    Vec3i operator+(const Vec3i &v)
    { return Vec3i(p[0]+v.p[0], p[1]+v.p[1], p[2]+v.p[2]); }
    Vec3i operator-(const Vec3i &v)
    { return Vec3i(p[0]-v.p[0], p[1]-v.p[1], p[2]-v.p[2]); }
    Vec3i operator-()
    { return Vec3i(-p[0], -p[1], -p[2]); }
    Vec3i operator*(int f)
    { return Vec3i(p[0]*f, p[1]*f, p[2]*f); }
    Vec3i operator/(int f)
    { return Vec3i(p[0]/f, p[1]/f, p[2]/f); }
    Vec3i& operator+=(const Vec3i& v)
    { p[0]+=v.p[0]; p[1]+=v.p[1]; p[2]+=v.p[2]; return *this; }
    Vec3i& operator*=(int f)
    { p[0]*=f; p[1]*=f; p[2]*=f; return *this; }
    Vec3i& operator/=(int f)
    { p[0]/=f; p[1]/=f; p[2]/=f; return *this; }
    bool operator==(const Vec3i &v)
    { return p[0]==v.p[0] && p[1]==v.p[1] && p[2]==v.p[2]; }
};

struct BBox3f
{
    Vec3f bmin, bmax;
    BBox3f::BBox3f(const Vec3f& a, const Vec3f& b)
    { bmin = a; bmax = b; }
    BBox3f::BBox3f() { }
};

class Vec4f
{
 private:
    float p[4];
 public:
    Vec4f() { p[0]=0.0; p[1]=0.0; p[2]=0.0;; p[3]=0.0; }
    Vec4f(float x, float y, float z, float w) { p[0]=x; p[1]=y; p[2]=z; p[3]=w; }
    Vec4f(const float *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; p[3]=f[3]; }
    Vec4f(double *d) { p[0]=d[0]; p[1]=d[1]; p[2]=d[2]; p[3]=d[3]; }
    Vec4f(const Vec4f& v) { p[0]=v.p[0]; p[1]=v.p[1]; p[2]=v.p[2]; p[3]=v.p[3];}
    inline void setValue(float x, float y, float z, float w) { p[0]=x; p[1]=y; p[2]=z; p[3]=w; }
    inline float* getValue() { return p; }
    inline void set(float *f) { p[0]=f[0]; p[1]=f[1]; p[2]=f[2]; p[3]=f[3]; }
    inline void scale(float s) { p[0]*=s; p[1]*=s; p[2]*=s; }
    inline double length() { return sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]+p[3]*p[3]); }
    inline void normalize() { scale(1.0/length()); }
    inline void setLength(double l) { scale(l/length()); }
    inline void negate() { p[0]=-p[0]; p[1]=-p[1]; p[2]=-p[2]; p[2]=-p[3]; }
    inline float& operator[](int n) { return p[n]; }
    
    Vec4f operator+(const Vec4f &v)
    { return Vec4f(p[0]+v.p[0], p[1]+v.p[1], p[2]+v.p[2], p[3]+v.p[3]); }
    Vec4f operator-(const Vec4f &v)
    { return Vec4f(p[0]-v.p[0], p[1]-v.p[1], p[2]-v.p[2], p[3]+v.p[3]); }
    Vec4f operator-()
    { return Vec4f(-p[0], -p[1], -p[2], -p[3]); }
    Vec4f operator*(double f)
    { return Vec4f(p[0]*f, p[1]*f, p[2]*f, p[3]*f); }
    Vec4f operator/(double f)
    { return Vec4f(p[0]/f, p[1]/f, p[2]/f, p[3]/f); }
    Vec4f& operator*=(double f)
    { p[0]*=f; p[1]*=f; p[2]*=f; p[3]*=f; return *this; }
    Vec4f& operator/=(double f)
    { p[0]/=f; p[1]/=f; p[2]/=f; p[3]/=f; return *this; }
    bool operator==(const Vec4f &v)
    { return fabs(p[0]-v.p[0])<DELTA
            && fabs(p[1]-v.p[1])<DELTA
            && fabs(p[2]-v.p[2])<DELTA
            && fabs(p[3]-v.p[3])<DELTA; }
    bool equals(const Vec4f &v, float tol)
    { return fabs(p[0]-v.p[0])<tol
            && fabs(p[1]-v.p[1])<tol
            && fabs(p[2]-v.p[2])<tol
            && fabs(p[3]-v.p[3])<tol; }
};

class Matrix
{
 private:
    float matrix[4][4];
    
 public:
    Matrix() {}
    Matrix(float* data);

    float* operator [](int i) { return &matrix[i][0]; }
    const float* operator [](int i) const  { return &matrix[i][0]; }
    
    void multVecMatrix(const Vec3f &src, Vec3f &dst) const;
    void multVecMatrix(const Vec3f &src, Vec4f &dst) const;
    void multMatrix(const Matrix &src, Matrix &dst) const;
};

class Rotation
{
 public:

    // Default constructor
    Rotation() {}

    // Constructor given a quaternion as an array of 4 components
    Rotation(const float v[4]) { setValue(v); }

    // Constructor given 4 individual components of a quaternion
    Rotation(float q0, float q1, float q2, float q3)
	{ setValue(q0, q1, q2, q3); }

    // Constructor given a rotation matrix
    Rotation(const Matrix &m) { setValue(m); }

    // Constructor given 3D rotation axis vector and angle in radians
    Rotation(const Vec3f &axis, float radians)
	{ setValue(axis, radians); }

    // Constructor for rotation that rotates one direction vector to another
    Rotation(const Vec3f &rotateFrom, const Vec3f &rotateTo)
	{ setValue(rotateFrom, rotateTo); }

    // Returns pointer to array of 4 components defining quaternion
    const float	* getValue() const { return (quat); }

    // Returns 4 individual components of rotation quaternion 
    void getValue(float &q0, float &q1, float &q2, float &q3) const;

    // Returns corresponding 3D rotation axis vector and angle in radians
    void getValue(Vec3f &axis, float &radians) const;

    // Returns corresponding 4x4 rotation matrix
    void getValue(Matrix &matrix) const;

    // Changes a rotation to be its inverse
    Rotation& invert();

    // Returns the inverse of a rotation
    Rotation inverse() const { Rotation q = *this; return q.invert(); }

    // Sets value of rotation from array of 4 components of a quaternion
    Rotation& setValue(const float q[4]);

    // Sets value of rotation from 4 individual components of a quaternion 
    Rotation& setValue(float q0, float q1, float q2, float q3);

    // Sets value of rotation from a rotation matrix
    Rotation& setValue(const Matrix &m);

    // Sets value of vector from 3D rotation axis vector and angle in radians
    Rotation& setValue(const Vec3f &axis, float radians);

    // Sets rotation to rotate one direction vector to another
    Rotation& setValue(const Vec3f &rotateFrom, const Vec3f &rotateTo);

    // Multiplies by another rotation; results in product of rotations
    Rotation& operator *=(const Rotation &q);

    // Equality comparison operator
    friend int operator ==(const Rotation &q1, const Rotation &q2);
    friend int operator !=(const Rotation &q1, const Rotation &q2)
	{ return !(q1 == q2); }

    // Equality comparison within given tolerance - the square of the
    // length of the maximum distance between the two quaternion vectors
    bool equals(const Rotation &r, float tolerance) const;

    // Multiplication of two rotations; results in product of rotations
    friend Rotation operator *(const Rotation &q1, const Rotation &q2);

    // Puts the given vector through this rotation
    // (Multiplies the given vector by the matrix of this rotation),.
    void multVec(const Vec3f &src, Vec3f &dst) const;

    // Keep the axis the same. Multiply the angle of rotation by 
    // the amount 'scaleFactor'
    void scaleAngle( float scaleFactor );

    // Spherical linear interpolation: as t goes from 0 to 1, returned
    // value goes from rot0 to rot1
    static Rotation slerp(const Rotation &rot0, const Rotation &rot1, float t);

    // Null rotation
    static Rotation identity()
	{ return Rotation(0.0, 0.0, 0.0, 1.0); }

 private:
    float quat[4]; // Storage for quaternion components

    // Returns the norm (square of the 4D length) of a rotation's quaterion
    float norm() const;

    // Normalizes a rotation quaternion to unit 4D length
    void normalize();
};

#endif

