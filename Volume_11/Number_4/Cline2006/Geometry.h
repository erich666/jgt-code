
//--------------------------------------------------------------------------//
//	Geometry.h - geometric objects needed for the ray tracer
//--------------------------------------------------------------------------//

#ifndef __GEOMETRY__
#define __GEOMETRY__

#include <vector>

#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define EPSILON  0.001f

double TIME(void);

class Point3;
class BoundingBox;
class Material;
class DirectionalLight;
class Ray;
class Intersection;
class Triangle;
class TriangleMesh;
class Scene;
class Camera;

class LBVH;
typedef LBVH* LBVHptr;

using namespace std;

//--------------------------------------------------------------------------//
//	Point3 - a point class made of 3 floats (also used for colors)
//--------------------------------------------------------------------------//

class Point3 
{
public:
	float A[3];

public:
	// CONSTRUCTORS
	Point3()  { A[0]=0.0f; A[1]=0.0f; A[2]=0.0f; }
	Point3(const float *X)  { A[0]=X[0]; A[1]=X[1]; A[2]=X[2]; }
	Point3(float x, float y, float z)  { A[0]=x; A[1]=y; A[2]=z; }
	Point3(const Point3 &P)  { A[0]=P[0]; A[1]=P[1]; A[2]=P[2]; }
	//
	~Point3() {}

	// SETTERS
	void set(float x, float y, float z)  { A[0]=x; A[1]=y; A[2]=z; }
	void set(Point3 &P) { A[0]=P[0]; A[1]=P[1]; A[2]=P[2]; }

	// MEMBER ACCESS FUNCTIONS
	float &operator[](int i) { return A[i]; }
	const float &operator[](int i) const { return A[i]; }
	float &x(void) { return A[0]; }
	float &y(void) { return A[1]; }
	float &z(void) { return A[2]; }

	// OPERATORS
	void         operator  =(const Point3 &P)  { A[0]=P[0]; A[1]=P[1]; A[2]=P[2]; }
	void         operator +=(const Point3 &P)  { A[0]+=P[0]; A[1]+=P[1]; A[2]+=P[2]; }
	const Point3 operator + (const Point3 &P) const  { Point3 S(*this); S+=P; return S; }
	void         operator -=(const Point3 &P)  { A[0]-=P[0]; A[1]-=P[1]; A[2]-=P[2]; }
	const Point3 operator - (const Point3 &P) const  { Point3 S(*this); S-=P; return S; }
	const Point3 operator - (void) const  { return Point3(-A[0], -A[1], -A[2]); }
	void         operator *=(float s)  { A[0]*=s; A[1]*=s; A[2]*=s; }
	const Point3 operator * (const float s) const  { Point3 P(*this); P*=s; return P; }
	friend const Point3 operator *(float s, const Point3 &P)  { Point3 Q(P); Q*=s; return Q; }
	const Point3 operator * (const Point3 &P) const  { Point3 Q(*this); Q*=P; return Q; }
	void         operator *=(const Point3 &P)  { A[0]*=P[0]; A[1]*=P[1]; A[2]*=P[2]; }

	// OTHER FUNCTIONS
	float length2(void) const { 
		return A[0]*A[0] + A[1]*A[1] + A[2]*A[2]; 
	}
	float length(void) const { 
		return sqrtf(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]); 
	}
	void  normalize(void) {
		float d = length();
		if (d > 0.0f) (*this) *= (1.0f/d);
	}
	float dot(const Point3 &P) const { 
		return( A[0]*P[0] + A[1]*P[1] + A[2]*P[2] ); 
	}
	const Point3 cross(const Point3 &P) const {
		return Point3( A[1]*P[2]-A[2]*P[1], A[2]*P[0]-A[0]*P[2], A[0]*P[1]-A[1]*P[0] );
	}
	const Point3 reflect(const Point3 &N) const {
		float NdotD = 2.0f * (N.dot(*this));
		return Point3( N[0]*NdotD - A[0], N[1]*NdotD - A[1], N[2]*NdotD - A[2] );
	}
	friend const Point3 minComponents(const Point3 &A, const Point3 &B) {
		return Point3( MIN(A[0],B[0]), MIN(A[1],B[1]), MIN(A[2],B[2]) );
	}
	friend const Point3 maxComponents(const Point3 &A, const Point3 &B) {
		return Point3( MAX(A[0],B[0]), MAX(A[1],B[1]), MAX(A[2],B[2]) );
	}
	void clamp(float min, float max) {
		A[0] = MIN(max, MIN(max, A[0]));
		A[1] = MIN(max, MIN(max, A[1]));
		A[2] = MIN(max, MIN(max, A[2]));
	}
	void print(void) {
		printf("(%0.4f, %0.4f, %0.4f)", A[0], A[1], A[2]);
	}
};

//--------------------------------------------------------------------------//
// BoundingBox - a floating point bounding box
//--------------------------------------------------------------------------//

class BoundingBox {
public:
	Point3 min, max;
	BoundingBox() {init();}
	~BoundingBox() {}

	void init(void) {
		min.set(FLT_MAX, FLT_MAX, FLT_MAX);  
		max.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	}
	void set(Point3 &minPoint, Point3 &maxPoint) {
		min = minPoint;
		max = maxPoint;
	}
	void expand(float x) {
		min -= Point3(x,x,x);
		max += Point3(x,x,x);
	}
	void expand(const Point3 &P) {
		min = minComponents(min,P); 
		max = maxComponents(max,P);
	}
	void expand(const BoundingBox &BB) {
		min = minComponents(min, BB.min); 
		max = maxComponents(max, BB.max);
	}
	const int getMajorAxis(void) const {
		Point3 P=max-min; 
		if (P[0]>=P[1] && P[0]>=P[2]) return 0; 
		if (P[1]>P[2]) return 1; 
		else return 2;
	}
	void print(void) {
		min.print(); 
		printf(" - "); 
		max.print();
	}
};

//--------------------------------------------------------------------------//
// Material - material properties
//--------------------------------------------------------------------------//

class Material
{
public:
	Point3 diffuse, specular;
	float exponent;

	Material() {
		diffuse.set(1,1,1);
		specular.set(0,0,0);
		exponent = 10.0f;
	}
	~Material() {}
};

//--------------------------------------------------------------------------//
// DirectionalLight
//--------------------------------------------------------------------------//

class DirectionalLight
{
public:
	Point3 direction;
	Point3 color;

	void set(const Point3 &d, const Point3 &c) { 
		color=c;
		direction=d;
		direction.normalize();
	}
};

//--------------------------------------------------------------------------//
// Ray - a ray (origin and direction)
//--------------------------------------------------------------------------//

class Ray
{
public:
	Point3 origin;
	Point3 direction;

	void set(const Point3 &o, const Point3 &d) {
		origin=o; 
		direction=d; 
		direction.normalize();
	}
};

//--------------------------------------------------------------------------//
// Intersection - encapsulates an intersection
//--------------------------------------------------------------------------//

class Intersection
{
public:
	float tval;
	Point3 location;
	Point3 normal;
	Material *material;

	void operator =(Intersection &I) {
		tval = I.tval; 
		location = I.location;
		normal = I.normal;
		material = I.material;
	}
	void set(float t, Point3 &loc, Point3 &norm, Material *mat) {
		tval = t;
		location = loc;
		normal = norm;
		material = mat;
	}

	const Point3 evaluateLighting(const Point3 &D, DirectionalLight &L) {
		Point3 val;
		float NdotL = normal.dot(L.direction);
		if (NdotL > 0.0f) {
			Point3 diffuse(L.color);
			diffuse *= material->diffuse;
			diffuse *= NdotL;
			val += diffuse;
			Point3 R = D.reflect(normal);
			float RdotL = R.dot(L.direction);
			if (RdotL > 0.0f) {
				Point3 specular(L.color);
				specular *= material->specular;
				specular *= powf(RdotL, material->exponent);
				val += specular;
			}
		}
		return val;
	}
};

//--------------------------------------------------------------------------//
// Triangle - a triangle used in conjunction with TriangleMesh
//--------------------------------------------------------------------------//

class Triangle 
{
public:
	int v0, v1, v2;

public:
	Triangle() {}
	~Triangle() {};

	void setVertices(int vert0, int vert1, int vert2) { v0=vert0; v1=vert1; v2=vert2; }
	void getNormal(Point3 &N, float a, float b, float c, TriangleMesh &mesh); 
	Point3 &vertex(int i, TriangleMesh &mesh); 
	void getBounds(BoundingBox &BB, TriangleMesh &mesh); 
	void operator =(const Triangle &T) { v0=T.v0; v1=T.v1; v2=T.v2; }
	void print(void) { printf("[%d, %d, %d]", v0,v1,v2); }

	bool intersectRay(Ray &ray, TriangleMesh &mesh, Intersection *intersection, float maxT);
};

//--------------------------------------------------------------------------//
// TriangleMesh - a goup of Triangles
//--------------------------------------------------------------------------//

class TriangleMesh
{
public:
	Material material;
	vector<Point3> vertices;
	vector<Point3> normals;
	//vector<Point3> texCoords; // not supporting textures
	vector<Triangle> triangles;

	TriangleMesh() {}
	~TriangleMesh() {}

	bool intersectTriangle(int i, Ray &ray, Intersection *intersection, float maxT) {
		return triangles[i].intersectRay(ray, *this, intersection, maxT);
	}
	void getBounds(BoundingBox &BB) {
		for (int i=(int)vertices.size()-1; i>=0; i--) BB.expand(vertices[i]);
	}
	void getTriangleBounds(BoundingBox &BB, int i) {
		if (i < (int)triangles.size()) triangles[i].getBounds(BB, *this);
		else BB.init();
	}
	int size(void) {
		return (int)triangles.size();
	}
	bool hasNormals(void) {
		return (normals.size() > 0);
	}
	void scale(Point3 &S) {
		for (int i=(int)vertices.size()-1; i>=0; i--) vertices[i]*=S;
	}
	void translate(Point3 &T) {
		for (int i=(int)vertices.size()-1; i>=0; i--) vertices[i]+=T;
	}
};

//--------------------------------------------------------------------------//
// Scene
//--------------------------------------------------------------------------//

class Scene
{
public:
	Point3 backgroundColor;
	Point3 ambientLight;
	vector<DirectionalLight*> lights;
	vector<LBVHptr> meshes;
	LBVH *rootBoundingVolume;

	Scene() {
		rootBoundingVolume=NULL;
		backgroundColor.set(0,0,0);
		ambientLight.set(0,0,0);
	}
	~Scene();
};

//--------------------------------------------------------------------------//
// Camera
//--------------------------------------------------------------------------//

class Camera
{
public:
	Point3 lookFrom;
	Point3 lookAt;
	Point3 viewUp;
	float fieldOfView;
	int imageWidth, imageHeight;
	Scene *scene;

	Camera() {
		lookFrom = Point3(0,0,10);
		lookAt   = Point3(0,0,0);
		viewUp   = Point3(0,1,0);
		fieldOfView = 0.5f;
		imageWidth = 256;
		imageHeight = 256;
	}
	~Camera() {}

	Point3 traceRay(Ray &ray);
	void captureImage(Scene *scene, char *imageName);
};

//--------------------------------------------------------------------------//
// Triangle function implementations
//--------------------------------------------------------------------------//

inline Point3& Triangle::vertex(int i, TriangleMesh &mesh)
{
	return mesh.vertices[(&v0)[i]];
}
//--------------------------------------------------------------------------//

inline void Triangle::getNormal(Point3 &N, float a, float b, float c, TriangleMesh &mesh)
{
	if (mesh.hasNormals()) {
		N = (a*mesh.normals[v0]) + (b*mesh.normals[v1]) + (c*mesh.normals[v2]);
		N.normalize();
	} else {
		N = (mesh.vertices[v1]-mesh.vertices[v0]).cross(mesh.vertices[v2]-mesh.vertices[v0]);
		N.normalize();
	}
}
//--------------------------------------------------------------------------//

inline void Triangle::getBounds(BoundingBox &BB, TriangleMesh &mesh)
{
	BB.set(mesh.vertices[v0], mesh.vertices[v0]);
	BB.expand(mesh.vertices[v1]);
	BB.expand(mesh.vertices[v2]);
}
//--------------------------------------------------------------------------//
	
#endif
