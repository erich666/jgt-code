/* trackball.c                               */
/* -----------                               */
/*                                           */
/* Code to implement a simple trackball-like */
/*     motion control.                       */
/*                                           */
/* This expands on Ed Angel's trackball.c    */
/*     demo program.  Though I think I've    */
/*     seen this code (trackball_ptov)       */
/*     before elsewhere.                     */
/*********************************************/


#include "glut_template.h"

GLint numTrackballs=-1;
GLint ballWidth = -1, ballHeight = -1;
GLfloat **lastPos; 
GLfloat **trackballMatrix; 
GLfloat **inverseTrackballMatrix; 

int matInvert(float src[16], float inverse[16]);

/* allocates memory for the correct number of trackballs */
void AllocateTrackballs( int totalNum )
{
	int i;
	lastPos = (GLfloat **)malloc( sizeof(GLfloat *) * totalNum );
	trackballMatrix = (GLfloat **)malloc( sizeof(GLfloat *) * totalNum );
	inverseTrackballMatrix = (GLfloat **)malloc( sizeof(GLfloat *) * totalNum );
	assert (lastPos && trackballMatrix && inverseTrackballMatrix);

	for (i=0; i<totalNum;i++)
	{
		lastPos[i] = (GLfloat *)malloc( sizeof(GLfloat) * 3 );
		trackballMatrix[i] = (GLfloat *)malloc( sizeof(GLfloat) * 16 );
		inverseTrackballMatrix[i] = (GLfloat *)malloc( sizeof(GLfloat) * 16 );
		assert( trackballMatrix[i] && lastPos[i] && inverseTrackballMatrix[i] );	
		lastPos[i][0] = lastPos[i][1] = lastPos[i][2] = 0;
		trackballMatrix[i][0] = trackballMatrix[i][5] = trackballMatrix[i][10] = trackballMatrix[i][15] = 1;
		trackballMatrix[i][1] = trackballMatrix[i][2] = trackballMatrix[i][3] = trackballMatrix[i][4] = 0;
		trackballMatrix[i][6] = trackballMatrix[i][7] = trackballMatrix[i][8] = trackballMatrix[i][9] = 0;
		trackballMatrix[i][11] = trackballMatrix[i][12] = trackballMatrix[i][13] = trackballMatrix[i][14] = 0;
		inverseTrackballMatrix[i][0] = inverseTrackballMatrix[i][5] = inverseTrackballMatrix[i][10] = inverseTrackballMatrix[i][15] = 1;
		inverseTrackballMatrix[i][1] = inverseTrackballMatrix[i][2] = inverseTrackballMatrix[i][3] = inverseTrackballMatrix[i][4] = 0;
		inverseTrackballMatrix[i][6] = inverseTrackballMatrix[i][7] = inverseTrackballMatrix[i][8] = inverseTrackballMatrix[i][9] = 0;
		inverseTrackballMatrix[i][11] = inverseTrackballMatrix[i][12] = inverseTrackballMatrix[i][13] = inverseTrackballMatrix[i][14] = 0;
	}
	numTrackballs = totalNum;
}

/* sets the size of the window the trackball assumes */
void ResizeTrackballWindow( int width, int height )
{
	ballWidth = width;
	ballHeight = height;
}

/* the internal code which computes the rotation */
void trackball_ptov(int x, int y, int width, int height, float v[3])
{
    float d, a;

    /* project x,y onto a hemi-sphere centered within width, height */
    v[0] = (2.0F*x - width) / width;
    v[1] = (height - 2.0F*y) / height;
    d = (float) sqrt(v[0]*v[0] + v[1]*v[1]);
    v[2] = (float) cos((M_PI/2.0F) * ((d < 1.0F) ? d : 1.0F));
    a = 1.0F / (float) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= a;
    v[1] *= a;
    v[2] *= a;
}

void SetTrackballOnClick( int ballNum, int x, int y )
{
	assert( numTrackballs > 0 && ballWidth > 0 && ballHeight > 0 );
	trackball_ptov( x, y, ballWidth, ballHeight, lastPos[ballNum] );
}

void UpdateTrackballOnMotion( int ballNum, int x, int y )
{
	float curPos[3], dx, dy, dz, angle, axis[3];
	//assert( numTrackballs > 0 && ballWidth > 0 && ballHeight > 0 );
	trackball_ptov( x, y, ballWidth, ballHeight, curPos );
    dx = curPos[0] - lastPos[ballNum][0];
	dy = curPos[1] - lastPos[ballNum][1];
	dz = curPos[2] - lastPos[ballNum][2];
	if ( fabs(dx) > 0 || fabs(dy) > 0 || fabs(dz) > 0 )
	{
		angle = 90 * sqrt( dx*dx + dy*dy + dz*dz );
		axis[0] = lastPos[ballNum][1]*curPos[2] - lastPos[ballNum][2]*curPos[1];
		axis[1] = lastPos[ballNum][2]*curPos[0] - lastPos[ballNum][0]*curPos[2];
		axis[2] = lastPos[ballNum][0]*curPos[1] - lastPos[ballNum][1]*curPos[0];
		lastPos[ballNum][0] = curPos[0];
		lastPos[ballNum][1] = curPos[1];
		lastPos[ballNum][2] = curPos[2];
		glPushMatrix();
		glLoadIdentity();
        glRotatef( angle, axis[0], axis[1], axis[2] );
		glMultMatrixf( trackballMatrix[ballNum] );
		glGetFloatv( GL_MODELVIEW_MATRIX, trackballMatrix[ballNum] );
		matInvert( trackballMatrix[ballNum], inverseTrackballMatrix[ballNum] );
		glPopMatrix();
	}
}

void MultiplyTrackballMatrix( int ballNum )
{
	glMultMatrixf( trackballMatrix[ballNum] );
}

void PrintTrackballMatrix( int ballNum )
{
	printf("Trackball %d Matrix:\n", ballNum );
	printf("%f %f %f %f\n", trackballMatrix[ballNum][0], trackballMatrix[ballNum][4], trackballMatrix[ballNum][8], trackballMatrix[ballNum][12] );
	printf("%f %f %f %f\n", trackballMatrix[ballNum][1], trackballMatrix[ballNum][5], trackballMatrix[ballNum][9], trackballMatrix[ballNum][13] );
	printf("%f %f %f %f\n", trackballMatrix[ballNum][2], trackballMatrix[ballNum][6], trackballMatrix[ballNum][10], trackballMatrix[ballNum][14] );
	printf("%f %f %f %f\n", trackballMatrix[ballNum][3], trackballMatrix[ballNum][7], trackballMatrix[ballNum][11], trackballMatrix[ballNum][15] );
}

void PrintInverseTrackballMatrix( int ballNum )
{
	printf("Trackball %d Matrix Inverse:\n", ballNum );
	printf("%f %f %f %f\n", inverseTrackballMatrix[ballNum][0], inverseTrackballMatrix[ballNum][4], inverseTrackballMatrix[ballNum][8], inverseTrackballMatrix[ballNum][12] );
	printf("%f %f %f %f\n", inverseTrackballMatrix[ballNum][1], inverseTrackballMatrix[ballNum][5], inverseTrackballMatrix[ballNum][9], inverseTrackballMatrix[ballNum][13] );
	printf("%f %f %f %f\n", inverseTrackballMatrix[ballNum][2], inverseTrackballMatrix[ballNum][6], inverseTrackballMatrix[ballNum][10], inverseTrackballMatrix[ballNum][14] );
	printf("%f %f %f %f\n", inverseTrackballMatrix[ballNum][3], inverseTrackballMatrix[ballNum][7], inverseTrackballMatrix[ballNum][11], inverseTrackballMatrix[ballNum][15] );
}

void MultiplyTransposeTrackballMatrix( int ballNum )
{
	glMultTransposeMatrixf( trackballMatrix[ballNum] );
}

void MultiplyInverseTrackballMatrix( int ballNum )
{
	glMultMatrixf( inverseTrackballMatrix[ballNum] );
}

void MultiplyInverseTransposeTrackballMatrix( int ballNum )
{
	glMultTransposeMatrixf( inverseTrackballMatrix[ballNum] );
}

void SetTrackballMatrixTo( int ballNum, GLfloat *newMat )
{
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixf( newMat );
	glGetFloatv( GL_MODELVIEW_MATRIX, trackballMatrix[ballNum] );
	matInvert( trackballMatrix[ballNum], inverseTrackballMatrix[ballNum] );
	glPopMatrix();
}

void GetTrackBallMatrix( int ballNum, GLfloat *matrix )
{
	int i;
	for (i=0;i<16;i++) matrix[i] = trackballMatrix[ballNum][i];
}


/***************************************************
** A couple matrix functions that are useful      **
***************************************************/


void matIdentity(float m[16])
{
    m[0+4*0] = 1; m[0+4*1] = 0; m[0+4*2] = 0; m[0+4*3] = 0;
    m[1+4*0] = 0; m[1+4*1] = 1; m[1+4*2] = 0; m[1+4*3] = 0;
    m[2+4*0] = 0; m[2+4*1] = 0; m[2+4*2] = 1; m[2+4*3] = 0;
    m[3+4*0] = 0; m[3+4*1] = 0; m[3+4*2] = 0; m[3+4*3] = 1;
}

void matIdentityd(double m[16])
{
    m[0+4*0] = 1; m[0+4*1] = 0; m[0+4*2] = 0; m[0+4*3] = 0;
    m[1+4*0] = 0; m[1+4*1] = 1; m[1+4*2] = 0; m[1+4*3] = 0;
    m[2+4*0] = 0; m[2+4*1] = 0; m[2+4*2] = 1; m[2+4*3] = 0;
    m[3+4*0] = 0; m[3+4*1] = 0; m[3+4*2] = 0; m[3+4*3] = 1;
}

void matTranspose(float res[16], float m[16])
{
  res[0] = m[0]; res[4] = m[1]; res[8] = m[2]; res[12] = m[3];
  res[1] = m[4]; res[5] = m[5]; res[9] = m[6]; res[13] = m[7];
  res[2] = m[8]; res[6] = m[9]; res[10] = m[10]; res[14] = m[11];
  res[3] = m[12]; res[7] = m[13]; res[11] = m[14]; res[15] = m[15];
}

void matTransposed(double res[16], double m[16])
{
  res[0] = m[0]; res[4] = m[1]; res[8] = m[2]; res[12] = m[3];
  res[1] = m[4]; res[5] = m[5]; res[9] = m[6]; res[13] = m[7];
  res[2] = m[8]; res[6] = m[9]; res[10] = m[10]; res[14] = m[11];
  res[3] = m[12]; res[7] = m[13]; res[11] = m[14]; res[15] = m[15];
}

int matInvert(float src[16], float inverse[16])
{
    float t;
    int i, j, k, swap;
    float tmp[4][4];

    matIdentity(inverse);

    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    tmp[i][j] = src[i*4+j];
	}
    }

    for (i = 0; i < 4; i++) {
        /* look for largest element in column. */
        swap = i;
        for (j = i + 1; j < 4; j++) {
            if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
                swap = j;
            }
        }

        if (swap != i) {
            /* swap rows. */
            for (k = 0; k < 4; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i*4+k];
                inverse[i*4+k] = inverse[swap*4+k];
                inverse[swap*4+k] = t;
            }
        }

        if (tmp[i][i] == 0) {
            /* no non-zero pivot.  the matrix is singular, which
	       shouldn't happen.  This means the user gave us a bad
	       matrix. */
            return 0;
        }

        t = tmp[i][i];
        for (k = 0; k < 4; k++) {
            tmp[i][k] /= t;
            inverse[i*4+k] /= t;
        }
        for (j = 0; j < 4; j++) {
            if (j != i) {
                t = tmp[j][i];
                for (k = 0; k < 4; k++) {
                    tmp[j][k] -= tmp[i][k]*t;
                    inverse[j*4+k] -= inverse[i*4+k]*t;
                }
            }
        }
    }
    return 1;
}

int matInvertd(double src[16], double inverse[16])
{
    double t;
    int i, j, k, swap;
    double tmp[4][4];

    matIdentityd(inverse);

    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    tmp[i][j] = src[i*4+j];
	}
    }

    for (i = 0; i < 4; i++) {
        /* look for largest element in column. */
        swap = i;
        for (j = i + 1; j < 4; j++) {
            if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
                swap = j;
            }
        }

        if (swap != i) {
            /* swap rows. */
            for (k = 0; k < 4; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i*4+k];
                inverse[i*4+k] = inverse[swap*4+k];
                inverse[swap*4+k] = t;
            }
        }

        if (tmp[i][i] == 0) {
            /* no non-zero pivot.  the matrix is singular, which
	       shouldn't happen.  This means the user gave us a bad
	       matrix. */
            return 0;
        }

        t = tmp[i][i];
        for (k = 0; k < 4; k++) {
            tmp[i][k] /= t;
            inverse[i*4+k] /= t;
        }
        for (j = 0; j < 4; j++) {
            if (j != i) {
                t = tmp[j][i];
                for (k = 0; k < 4; k++) {
                    tmp[j][k] -= tmp[i][k]*t;
                    inverse[j*4+k] -= inverse[i*4+k]*t;
                }
            }
        }
    }
    return 1;
}


void matInverseTranspose( float res[16], float m[16] )
{
  float inv[16];
  matInvert( m, inv );
  matTranspose( res, inv );
}

void matInverseTransposed( double res[16], double m[16] )
{
  double inv[16];
  matInvertd( m, inv );
  matTransposed( res, inv );
}