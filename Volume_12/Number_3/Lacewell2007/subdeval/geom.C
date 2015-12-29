#include "geom.h"

void Vec3f::rotateXY(float theta, float tx, float ty)
{
    float costh = cos(theta*RADDEG);
    float sinth = sin(theta*RADDEG);
    float x = p[0]*costh-p[1]*sinth+(-tx*costh+ty*sinth+tx);
    float y = p[0]*sinth+p[1]*costh+(-tx*sinth-ty*costh+ty);
    p[0] = x;
    p[1] = y;
}


void Matrix::multVecMatrix(const Vec3f &src, Vec3f &dst) const
{
    float	x,y,z,w;
    
    x = src[0]*matrix[0][0] + src[1]*matrix[1][0] +
	src[2]*matrix[2][0] + matrix[3][0];
    y = src[0]*matrix[0][1] + src[1]*matrix[1][1] +
	src[2]*matrix[2][1] + matrix[3][1];
    z = src[0]*matrix[0][2] + src[1]*matrix[1][2] +
	src[2]*matrix[2][2] + matrix[3][2];
    w = src[0]*matrix[0][3] + src[1]*matrix[1][3] +
	src[2]*matrix[2][3] + matrix[3][3];
    
    dst.setValue(x/w, y/w, z/w);
}


void Matrix::multVecMatrix(const Vec3f &src, Vec4f &dst) const
{
    float	x,y,z,w;
    
    x = src[0]*matrix[0][0] + src[1]*matrix[1][0] +
	src[2]*matrix[2][0] + matrix[3][0];
    y = src[0]*matrix[0][1] + src[1]*matrix[1][1] +
	src[2]*matrix[2][1] + matrix[3][1];
    z = src[0]*matrix[0][2] + src[1]*matrix[1][2] +
	src[2]*matrix[2][2] + matrix[3][2];
    w = src[0]*matrix[0][3] + src[1]*matrix[1][3] +
	src[2]*matrix[2][3] + matrix[3][3];
    
    dst.setValue(x, y, z, w);
}


void Matrix::multMatrix(const Matrix &src, Matrix &dst) const
{
    const float* a = &matrix[0][0];
    const float* b = src[0];
    float* c = dst[0];

    c[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]  + a[3]*b[12];
    c[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]  + a[3]*b[13];
    c[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
    c[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];

    c[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]  + a[7]*b[12];
    c[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]  + a[7]*b[13];
    c[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
    c[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];

    c[8]  = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]  + a[11]*b[12];
    c[9]  = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]  + a[11]*b[13];
    c[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10] + a[11]*b[14];
    c[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]*b[15];

    c[12] = a[12]*b[0] + a[13]*b[4] + a[14]*b[8]  + a[15]*b[12];
    c[13] = a[12]*b[1] + a[13]*b[5] + a[14]*b[9]  + a[15]*b[13];
    c[14] = a[12]*b[2] + a[13]*b[6] + a[14]*b[10] + a[15]*b[14];
    c[15] = a[12]*b[3] + a[13]*b[7] + a[14]*b[11] + a[15]*b[15];
}



//    Returns 4 individual components of rotation quaternion.
void Rotation::getValue(float &q0, float &q1, float &q2, float &q3) const
{
    q0 = quat[0];
    q1 = quat[1];
    q2 = quat[2];
    q3 = quat[3];
}

//    Returns corresponding 3D rotation axis vector and angle in radians.
void Rotation::getValue(Vec3f &axis, float &radians) const
{
    float	len;
    Vec3f	q;

    q[0] = quat[0];
    q[1] = quat[1];
    q[2] = quat[2];

    if ((len = q.length()) > 0.00001) {
	axis	= q * (1.0 / len);
	radians	= 2.0 * acosf(quat[3]);
    }

    else {
	axis.setValue(0.0, 0.0, 1.0);
	radians = 0.0;
    }
}

//    Returns corresponding 4x4 rotation matrix.
void Rotation::getValue(Matrix &matrix) const
{
    Matrix m;

    m[0][0] = 1 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]);
    m[0][1] =     2.0 * (quat[0] * quat[1] + quat[2] * quat[3]);
    m[0][2] =     2.0 * (quat[2] * quat[0] - quat[1] * quat[3]);
    m[0][3] = 0.0;

    m[1][0] =     2.0 * (quat[0] * quat[1] - quat[2] * quat[3]);
    m[1][1] = 1 - 2.0 * (quat[2] * quat[2] + quat[0] * quat[0]);
    m[1][2] =     2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    m[1][3] = 0.0;

    m[2][0] =     2.0 * (quat[2] * quat[0] + quat[1] * quat[3]);
    m[2][1] =     2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    m[2][2] = 1 - 2.0 * (quat[1] * quat[1] + quat[0] * quat[0]);
    m[2][3] = 0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;

    matrix = m;
}

//    Changes a rotation to be its inverse.
Rotation& Rotation::invert()
{
    float invNorm = 1.0 / norm();

    quat[0] = -quat[0] * invNorm;
    quat[1] = -quat[1] * invNorm;
    quat[2] = -quat[2] * invNorm;
    quat[3] =  quat[3] * invNorm;

    return *this;
}

//    Sets value of rotation from array of 4 components of a
//    quaternion.
Rotation& Rotation::setValue(const float q[4])
{
    quat[0] = q[0];
    quat[1] = q[1];
    quat[2] = q[2];
    quat[3] = q[3];
    normalize();

    return (*this);
}

//    Sets value of rotation from 4 individual components of a
//    quaternion.
Rotation& Rotation::setValue(float q0, float q1, float q2, float q3)
{
    quat[0] = q0;
    quat[1] = q1;
    quat[2] = q2;
    quat[3] = q3;
    normalize();

    return (*this);
}

//    Sets value of rotation from a rotation matrix.
Rotation& Rotation::setValue(const Matrix &m)
{
    int i, j, k;

    // First, find largest diagonal in matrix:
    if (m[0][0] > m[1][1]) {
	if (m[0][0] > m[2][2]) {
	    i = 0;
	}
	else i = 2;
    }
    else {
	if (m[1][1] > m[2][2]) {
	    i = 1;
	}
	else i = 2;
    }
    if (m[0][0]+m[1][1]+m[2][2] > m[i][i]) {
	// Compute w first:
	quat[3] = sqrt(m[0][0]+m[1][1]+m[2][2]+m[3][3])/2.0;

	// And compute other values:
	quat[0] = (m[1][2]-m[2][1])/(4*quat[3]);
	quat[1] = (m[2][0]-m[0][2])/(4*quat[3]);
	quat[2] = (m[0][1]-m[1][0])/(4*quat[3]);
    }
    else {
	// Compute x, y, or z first:
	j = (i+1)%3; k = (i+2)%3;
    
	// Compute first value:
	quat[i] = sqrt(m[i][i]-m[j][j]-m[k][k]+m[3][3])/2.0;
       
	// And the others:
	quat[j] = (m[i][j]+m[j][i])/(4*quat[i]);
	quat[k] = (m[i][k]+m[k][i])/(4*quat[i]);

	quat[3] = (m[j][k]-m[k][j])/(4*quat[i]);
    }
    
    return (*this);
}

//    Sets value of rotation from 3D rotation axis vector and angle in
//    radians.
Rotation& Rotation::setValue(const Vec3f &axis, float radians)
{
    Vec3f	q;

    q = axis;
    q.normalize();

    q *= sinf(radians / 2.0);

    quat[0] = q[0];
    quat[1] = q[1];
    quat[2] = q[2];

    quat[3] = cosf(radians / 2.0);

    return(*this);
}

//    Sets rotation to rotate one direction vector to another.
Rotation& Rotation::setValue(const Vec3f &rotateFrom, const Vec3f &rotateTo)
{
    Vec3f	from = rotateFrom;
    Vec3f	to = rotateTo;
    Vec3f	axis;
    float	cost;

    from.normalize();
    to.normalize();
    cost = from.dot(to);

    // check for degeneracies
    if (cost > 0.99999) {		// vectors are parallel
	quat[0] = quat[1] = quat[2] = 0.0;
	quat[3] = 1.0;
	return *this;
    }
    else if (cost < -0.99999) {		// vectors are opposite
	// find an axis to rotate around, which should be
	// perpendicular to the original axis
	// Try cross product with (1,0,0) first, if that's one of our
	// original vectors then try  (0,1,0).
	Vec3f tmp = from.cross(Vec3f(1.0, 0.0, 0.0));
	if (tmp.length() < 0.00001)
	    tmp = from.cross(Vec3f(0.0, 1.0, 0.0));

	tmp.normalize();
	setValue(tmp[0], tmp[1], tmp[2], 0.0);
	return *this;
    }

    axis = rotateFrom.cross(rotateTo);
    axis.normalize();

    // use half-angle formulae
    // sin^2 t = ( 1 - cos (2t) ) / 2
    axis *= sqrt(0.5 * (1.0 - cost));

    // scale the axis by the sine of half the rotation angle to get
    // the normalized quaternion
    quat[0] = axis[0];
    quat[1] = axis[1];
    quat[2] = axis[2];

    // cos^2 t = ( 1 + cos (2t) ) / 2
    // w part is cosine of half the rotation angle
    quat[3] = sqrt(0.5 * (1.0 + cost));

    return (*this);
}

//    Multiplies by another rotation.
Rotation& Rotation::operator *=(const Rotation &q)
{
    float p0, p1, p2, p3;

    p0 = (q.quat[3] * quat[0] + q.quat[0] * quat[3] +
	  q.quat[1] * quat[2] - q.quat[2] * quat[1]);
    p1 = (q.quat[3] * quat[1] + q.quat[1] * quat[3] +
	  q.quat[2] * quat[0] - q.quat[0] * quat[2]);
    p2 = (q.quat[3] * quat[2] + q.quat[2] * quat[3] +
	  q.quat[0] * quat[1] - q.quat[1] * quat[0]);
    p3 = (q.quat[3] * quat[3] - q.quat[0] * quat[0] -
	  q.quat[1] * quat[1] - q.quat[2] * quat[2]);
    quat[0] = p0;
    quat[1] = p1;
    quat[2] = p2;
    quat[3] = p3;

    normalize();

    return(*this);
}

//    Equality comparison operator.
int operator ==(const Rotation &q1, const Rotation &q2)
{
    return (q1.quat[0] == q2.quat[0] &&
	    q1.quat[1] == q2.quat[1] &&
	    q1.quat[2] == q2.quat[2] &&
	    q1.quat[3] == q2.quat[3]);
}

//    Equality comparison operator within given tolerance - the square
//    of the length of the maximum distance between the two vectors.

bool Rotation::equals(const Rotation &r, float tolerance) const
{
    return Vec4f(quat).equals(Vec4f(r.quat), tolerance);
}

//    Binary multiplication operator.

Rotation operator *(const Rotation &q1, const Rotation &q2)
{
    Rotation q(q2.quat[3] * q1.quat[0] + q2.quat[0] * q1.quat[3] +
		  q2.quat[1] * q1.quat[2] - q2.quat[2] * q1.quat[1],

		  q2.quat[3] * q1.quat[1] + q2.quat[1] * q1.quat[3] +
		  q2.quat[2] * q1.quat[0] - q2.quat[0] * q1.quat[2],

		  q2.quat[3] * q1.quat[2] + q2.quat[2] * q1.quat[3] +
		  q2.quat[0] * q1.quat[1] - q2.quat[1] * q1.quat[0],

		  q2.quat[3] * q1.quat[3] - q2.quat[0] * q1.quat[0] -
		  q2.quat[1] * q1.quat[1] - q2.quat[2] * q1.quat[2]);
    q.normalize();

    return (q);
}

//    Puts the given vector through this rotation
//    (Multiplies the given vector by the matrix of this rotation).
void Rotation::multVec(const Vec3f &src, Vec3f &dst) const
{
    Matrix myMat;
    getValue( myMat );

    myMat.multVecMatrix( src, dst );
}

//    Keep the axis the same. Multiply the angle of rotation by
//    the amount 'scaleFactor'

void Rotation::scaleAngle(float scaleFactor )
{
    Vec3f myAxis;
    float   myAngle;

    // Get the Axis and angle.
    getValue( myAxis, myAngle );

    setValue( myAxis, (myAngle * scaleFactor) );
}

//    Spherical linear interpolation: as t goes from 0 to 1, returned
//    value goes from rot0 to rot1.
Rotation Rotation::slerp(const Rotation &rot0, const Rotation &rot1, float t)
{
        const float*    r1q = rot1.getValue();

        Rotation      rot;
        float           rot1q[4];
        double          omega, cosom, sinom;
        double          scalerot0, scalerot1;
        int             i;

        // Calculate the cosine
        cosom = rot0.quat[0]*rot1.quat[0] + rot0.quat[1]*rot1.quat[1]
                + rot0.quat[2]*rot1.quat[2] + rot0.quat[3]*rot1.quat[3];

        // adjust signs if necessary
        if ( cosom < 0.0 ) {
                cosom = -cosom;
                for ( int j = 0; j < 4; j++ )
                        rot1q[j] = -r1q[j];
        } else  {
                for ( int j = 0; j < 4; j++ )
                        rot1q[j] = r1q[j];
        }

        // calculate interpolating coeffs
        if ( (1.0 - cosom) > 0.00001 ) {
                // standard case
                omega = acos(cosom);
                sinom = sin(omega);
                scalerot0 = sin((1.0 - t) * omega) / sinom;
                scalerot1 = sin(t * omega) / sinom;
        } else {        
                // rot0 and rot1 very close - just do linear interp.
                scalerot0 = 1.0 - t;
                scalerot1 = t;
        }

        // build the new quarternion
        for (i = 0; i <4; i++)
                rot.quat[i] = scalerot0 * rot0.quat[i] + scalerot1 * rot1q[i];

        return rot;
}

//    Returns the norm (square of the 4D length) of the quaternion
//    defining the rotation.
float Rotation::norm() const
{
    return (quat[0] * quat[0] +
	    quat[1] * quat[1] +
	    quat[2] * quat[2] +
	    quat[3] * quat[3]);
}

//    Normalizes a rotation quaternion to unit 4D length.
//
// Use: private

void Rotation::normalize()
{
    float	dist = 1.0 / sqrt(norm());

    quat[0] *= dist;
    quat[1] *= dist;
    quat[2] *= dist;
    quat[3] *= dist;
}

