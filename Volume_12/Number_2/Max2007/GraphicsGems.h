/* A modified version of the graphics gems header file.
 * Main changes: - C89 function definitions and full decarations in this file for
 *                 the vector library (Daniel Laney).
 *               - Added comment on the convention assumed for the standard
 *                 transformation matrices . 
 *               - Added new point transformation for 3-D based on premultiplication
 *                 of column vectors by matrices.
 * GraphicsGems.h
 * Version 1.0 - Andrew Glassner
 * from "Graphics Gems", Academic Press, 1990
EULA: The Graphics Gems code is copyright-protected. In other words, you cannot claim the
text of the code as your own and resell it. Using the code is permitted in any program, 
product, or library, non-commercial or commercial. Giving credit is not required, though 
is a nice gesture. The code comes as-is, and if there are any flaws or problems with any 
Gems code, nobody involved with Gems - authors, editors, publishers, or webmasters - are 
to be held responsible. Basically, don't be a jerk, and remember that anything free comes 
with no guarantee.
 */

#ifndef GG_H

#define GG_H 1

#define GGVEC_PREMULTIPLY 1

/*********************/
/* 2d geometry types */
/*********************/

typedef struct Point2Struct {   /* 2d point */
    double x, y;
    } Point2;
typedef Point2 Vector2;

typedef struct IntPoint2Struct {    /* 2d integer point */
    int x, y;
    } IntPoint2;

/* Added by Daniel Laney, see comment for M4ELT */
#define M3ELT(m, row, col) ((m).element[row][col])

typedef struct Matrix3Struct {  /* 3-by-3 matrix */
    double element[3][3];
    } Matrix3;

typedef struct Box2dStruct {        /* 2d box */
    Point2 min, max;
    } Box2;
    

/*********************/
/* 3d geometry types */
/*********************/

typedef struct Point3Struct {   /* 3d point */
    double x, y, z;
    } Point3;
typedef Point3 Vector3;

typedef struct IntPoint3Struct {    /* 3d integer point */
    int x, y, z;
    } IntPoint3;


/* Added by Daniel Laney to clarify how the matrix rows and columns
 * are indexed in the Matrix4Struct.  This does not effect the
 * conventions used to interpret the matrices (for example, whether
 * we mulitply a row vector into a matrix, or a matrix into a column
 * vector to transform a point).
 */
#define M4ELT(m, row, col) ((m).element[row][col])

typedef struct Matrix4Struct {  /* 4-by-4 matrix */
    double element[4][4];
    } Matrix4;

typedef struct Box3dStruct {        /* 3d box */
    Point3 min, max;
    } Box3;



/***********************/
/* one-argument macros */
/***********************/

/* absolute value of a */
#define ABS(a)      (((a)<0) ? -(a) : (a))

/* round a to nearest int */
#define ROUND(a)    ((a)>0 ? (int)((a)+0.5) : -(int)(0.5-(a)))

/* take sign of a, either -1, 0, or 1 */
#define ZSGN(a)     (((a)<0) ? -1 : (a)>0 ? 1 : 0)  

/* take binary sign of a, either -1, or 1 if >= 0 */
#define SGN(a)      (((a)<0) ? -1 : 1)

/* shout if something that should be true isn't */
#define ASSERT(x) \
if (!(x)) fprintf(stderr," Assert failed: x\n");

/* square a */
#define SQR(a)      ((a)*(a))   


/***********************/
/* two-argument macros */
/***********************/

/* find minimum of a and b */
#define MIN(a,b)    (((a)<(b))?(a):(b)) 

/* find maximum of a and b */
#define MAX(a,b)    (((a)>(b))?(a):(b)) 

/* swap a and b (see Gem by Wyvill) */
#define SWAP(a,b)   { a^=b; b^=a; a^=b; }

/* linear interpolation from l (when a=0) to h (when a=1)*/
/* (equal to (a*h)+((1-a)*l) */
#define LERP(a,l,h) ((l)+(((h)-(l))*(a)))

/* clamp the input to the specified range */
#define CLAMP(v,l,h)    ((v)<(l) ? (l) : (v) > (h) ? (h) : v)


/****************************/
/* memory allocation macros */
/****************************/

/* create a new instance of a structure (see Gem by Hultquist) */
#define NEWSTRUCT(x)    (struct x *)(malloc((unsigned)sizeof(struct x)))

/* create a new instance of a type */
#define NEWTYPE(x)  (x *)(malloc((unsigned)sizeof(x)))


/********************/
/* useful constants */
/********************/

#define PI      3.141592    /* the venerable pi */
#define PITIMES2    6.283185    /* 2 * pi */
#define PIOVER2     1.570796    /* pi / 2 */
#define E       2.718282    /* the venerable e */
#define SQRT2       1.414214    /* sqrt(2) */
#define SQRT3       1.732051    /* sqrt(3) */
#define GOLDEN      1.618034    /* the golden ratio */
#define DTOR        0.017453    /* convert degrees to radians */
#define RTOD        57.29578    /* convert radians to degrees */


/************/
/* booleans */
/************/

typedef int boolean;            /* boolean data type */
typedef boolean flag;           /* flag data type */


/******************/
/*   2d Library   */
/******************/

/* returns squared length of input vector */    
double V2SquaredLength(const Vector2* a);

/* returns length of input vector */
double V2Length(const Vector2* a);

/* negates the input vector and returns it */
Vector2* V2Negate(Vector2* v);

/* normalizes the input vector and returns it */
Vector2* V2Normalize(Vector2* v);

/* scales the input vector to the new length and returns it */
Vector2 *V2Scale(Vector2* v, double newlen);

/* return vector sum c = a+b */
Vector2 *V2Add(const Vector2* a, const Vector2* b, Vector2* c);

/* return vector difference c = a-b */
Vector2 *V2Sub(const Vector2* a, const Vector2* b, Vector2* c);

/* return the dot product of vectors a and b */
double V2Dot(const Vector2* a, const Vector2* b);

/* linearly interpolate between vectors by an amount alpha */
/* and return the resulting vector. */
/* When alpha=0, result=lo.  When alpha=1, result=hi. */
Vector2 *V2Lerp(const Vector2* lo, const Vector2* hi, double alpha, Vector2* result);

/* make a linear combination of two vectors and return the result. */
/* result = (a * ascl) + (b * bscl) */
Vector2 *V2Combine (const Vector2* a, const Vector2* b, Vector2* result, 
                    double ascl, double bscl);

/* multiply two vectors together component-wise */
Vector2 *V2Mul (const Vector2* a, const Vector2* b, Vector2* result);

/* return the distance between two points */
double V2DistanceBetween2Points(const Point2* a, const Point2* b);

/* return the vector perpendicular to the input vector a */
Vector2 *V2MakePerpendicular(const Vector2* a, Vector2* ap);

/* create, initialize, and return a new vector */
Vector2 *V2New(double x, double y);

/* create, initialize, and return a duplicate vector */
Vector2 *V2Duplicate(const Vector2* a);

/* ------------------------------------------------------------
 * V3MulPointByMatrix()
 * ------------------------------------------------------------
 *
 * Multiply a point by a matrix and return the transformed point.
 * Assumes a post multiply operation.  That is, we multiply a row vector on the
 * left into the matrix (for 2-D):
 *  [x y 1] |e00 e01 e02|
 *          |e10 e11 e12| = [x'/w  y'/w' 1]
 *          |e20 e21 e22|
 *
 * The elements of the result are automatically divided by the
 * assumed homogeneous coordinate if it is non-zero.
 */
Point2 *V2MulPointByMatrix(Point2* p, const Matrix3* m);

/*
 * Standard transform by multiplying a column vector 'p' on the left
 * by the matrix 'm', resulting in a column vector 'q'.
 */
Point2*
V2MatVecMul(const Matrix3* m, const Point2* p, Point2* q);

/* multiply together matrices c = ab */
/* note that c must not point to either of the input matrices */
Matrix3 *V2MatMul(const Matrix3* a, const Matrix3* b, Matrix3* c);


/******************/
/*   3d Library   */
/******************/

/*
 * Standard transform by multiplying a column vector 'p' on the left
 * by the matrix 'm', resulting in a column vector 'q' and a homogeneous coordinate 'w'.
 * The components of 'q' must be divided by 'w' to get the final coordinates, this
 * macro does not do this.
 */
#define V3MATVECMUL(m, p, q, w) \
{ \
    (q).x = ((p).x * (m).element[0][0]) + ((p).y * (m).element[0][1]) + ((p).z * (m).element[0][2]) + (m).element[0][3]; \
    (q).y = ((p).x * (m).element[1][0]) + ((p).y * (m).element[1][1]) + ((p).z * (m).element[1][2]) + (m).element[1][3]; \
    (q).z = ((p).x * (m).element[2][0]) + ((p).y * (m).element[2][1]) + ((p).z * (m).element[2][2]) + (m).element[2][3]; \
    w     = ((p).x * (m).element[3][0]) + ((p).y * (m).element[3][1]) + ((p).z * (m).element[3][2]) + (m).element[3][3]; \
}

/*
 * This version does not compute the homogeneous coordinate.
 */
#define V3MATVECMUL2(m, p, q) \
{ \
    (q).x = ((p).x * (m).element[0][0]) + ((p).y * (m).element[0][1]) + ((p).z * (m).element[0][2]) + (m).element[0][3]; \
    (q).y = ((p).x * (m).element[1][0]) + ((p).y * (m).element[1][1]) + ((p).z * (m).element[1][2]) + (m).element[1][3]; \
    (q).z = ((p).x * (m).element[2][0]) + ((p).y * (m).element[2][1]) + ((p).z * (m).element[2][2]) + (m).element[2][3]; \
}

void
V3Set(Vector3* v, double x, double y, double c);

/* returns squared length of input vector */    
double V3SquaredLength(const Vector3* a);

/* returns length of input vector */
double V3Length(const Vector3* a);

/* negates the input vector and returns it */
Vector3 *V3Negate(Vector3* v);

/* normalizes the input vector and returns it */
Vector3 *V3Normalize(Vector3* v);

/* scales the input vector to the new length and returns it */
Vector3 *V3Scale(Vector3* v, double newlen);

Vector3* V3Copy(const Vector3* src, Vector3* dest);

/* return vector sum c = a+b */
Vector3 *V3Add(const Vector3* a, const Vector3* b, Vector3* c);

/* return vector difference c = a-b */
Vector3 *V3Sub(const Vector3* a, const Vector3* b, Vector3* c);

/* return the dot product of vectors a and b */
double V3Dot(const Vector3* a, const Vector3* b);

/* linearly interpolate between vectors by an amount alpha */
/* and return the resulting vector. */
/* When alpha=0, result=lo.  When alpha=1, result=hi. */
Vector3 *V3Lerp(const Vector3* lo, const Vector3* hi, double alpha, 
                Vector3* result);

/* make a linear combination of two vectors and return the result. */
/* result = (a * ascl) + (b * bscl) */
Vector3 *V3Combine (const Vector3* a, const Vector3* b, Vector3* result, 
                    double ascl, double bscl);

/* multiply two vectors together component-wise and return the result */
Vector3 *V3Mul (const Vector3* a, const Vector3* b, Vector3* result);

/* return the distance between two points */
double V3DistanceBetween2Points(const Point3* a, const Point3* b);

/* return the cross product c = a cross b */
Vector3 *V3Cross(const Vector3* a, const Vector3* b, Vector3* c);

/* create, initialize, and return a new vector */
Vector3 *V3New(double x, double y, double z);

/* create, initialize, and return a duplicate vector */
Vector3 *V3Duplicate(const Vector3* a);

/* Added by Daniel Laney */
void
M4LoadIdentity(Matrix4* m);

/* Added by Daniel Laney */
void
M4Premultiply(Matrix4* result, const Matrix4* a);

/* ------------------------------------------------------------
 * V3MulPointByMatrix()
 * ------------------------------------------------------------
 *
 * Multiply a point by a matrix and return the transformed point.
 * Assumes a POST multiply operation.  That is, we multiply a row vector on the
 * left into the matrix (for 2-D):
 *  [x y 1] |e00 e01 e02|
 *          |e10 e11 e12| = [x'/w  y'/w  1]
 *          |e20 e21 e22|
 *
 * The elements of the result are automatically divided by the
 * assumed homogeneous coordinate if it is non-zero.
 *
 */

Point3 *V3MulPointByMatrix(Point3* p, const Matrix4* m);

/* ------------------------------------------------------------
 * V3MatVecMul()
 * ------------------------------------------------------------
 *
 * Multiply a point by a matrix and return the transformed point.
 * The input point 'p' is assumed to have an implicit homogeneous
 * coordinate of 1.
 * 'q' is updated to the transformed point, and also returned.
 * Assumes a PRE multiply operation.  That is, we multiply a column vector
 * by the matrix (for 2-D):
 *          |e00 e01 e02| |x|   |x'/w|
 *          |e10 e11 e12| |y| = |y'/w|
 *          |e20 e21 e22| |1|   |1 |
 *
 * The elements of the result are automatically divided by the
 * assumed homogeneous coordinate if it is non-zero.
 * New routine, added by Daniel Laney.
 */
Point3 *V3MatVecMul(const Matrix4* m, const Point3* p, Point3* q);

/* multiply together matrices c = ab */
/* note that c must not point to either of the input matrices */
Matrix4 *V3MatMul(const Matrix4* a, const Matrix4* b, Matrix4* c);


/***********************/
/*   Useful Routines   */
/***********************/

/* binary greatest common divisor by Silver and Terzian.  See Knuth */
/* both inputs must be >= 0 */
int
gcd(int u, int v);

/* return roots of ax^2+bx+c */
/* stable algebra derived from Numerical Recipes by Press et al.*/
int quadraticRoots(double a, double b, double c, double* roots);


/* generic 1d regula-falsi step.  f is function to evaluate */
/* interval known to contain root is given in left, right */
/* returns new estimate */
double RegulaFalsi(double (*f)(), double left, double right);

/* generic 1d Newton-Raphson step. f is function, df is derivative */
/* x is current best guess for root location. Returns new estimate */
double NewtonRaphson(double (*f)(), double (*df)(), double x);

/* hybrid 1d Newton-Raphson/Regula Falsi root finder. */
/* input function f and its derivative df, an interval */
/* left, right known to contain the root, and an error tolerance */
/* Based on Blinn */
double findroot(double left, double right, double tolerance, 
                double (*f)(), double (*df)());


/* RETURN the parameter 't' such that:
 * p = p0 + t(p1-p0) is the intersection point of the line segment p0-p1 and plane v0-v1-v2 
 * t < 0 or t > 1 means the line segment does not intersect the plane, but the line
 * of infinite extent containing the segment does intersect the plane.
 * This routine is copied from xform.c, it is not originally part of the
 * graphics gems library.
 */
double
P3LinePlaneIntersection(const Point3* p0, const Point3* p1, 
                           const Point3* v0, const Point3* v1, const Point3* v2);

void
V3Print(const Vector3* v);

void
M4Print(const Matrix4* m);

#endif
