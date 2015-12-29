/*****************************************
** materials.c                          **
** -----------                          **
**                                      **
** This file contains assorted material **
** types for OpenGL when used in a lit  **
** environment.  Examples: brass, gold  **
** plastic, silver, etc.                **
**                                      **
** Chris Wyman (2/15/2000)              **
*****************************************/

#include <GL/glut.h>
#include "materials.h"

/* material properties structure */
typedef struct {
  GLfloat ambient[4];
  GLfloat diffuse[4];
  GLfloat specular[4];
  GLfloat shiny;
} _mat_prop;


/* list of materials */
_mat_prop mat[] = {
  { { 0.329412, 0.223529, 0.027451, 1.0},
    { 0.780392, 0.568627, 0.113725, 1.0},
    { 0.992157, 0.941176, 0.807843, 1.0},
    27.8974
  },
  { { 0.2125, 0.1275, 0.054, 1.0},
    { 0.714, 0.4284, 0.18144, 1.0},
    { 0.393548, 0.271906, 0.166721, 1.0},
    25.6
  },
  { { 0.25, 0.148, 0.06475, 1.0},
    { 0.4, 0.2368, 0.1036, 1.0},
    { 0.774597, 0.458561, 0.200621, 1.0},
    76.8
  },
  { { 0.25, 0.25, 0.25, 1.0},
    { 0.4, 0.4, 0.4, 1.0},
    { 0.774597, 0.774597, 0.774597, 1.0},
    76.8
  },
  { { 0.19125, 0.0735, 0.0225, 1.0},
    { 0.7038, 0.27048, 0.0828, 1.0},
    { 0.256777, 0.137622, 0.086014, 1.0},
    12.8
  },
  { { 0.2295, 0.08825, 0.0275, 1.0},
    { 0.5508, 0.2118, 0.066, 1.0},
    { 0.580594, 0.223257, 0.0695701, 1.0},
    51.2
  },
  { { 0.24725, 0.1995, 0.0745, 1.0},
    { 0.75164, 0.60648, 0.22648, 1.0},
    { 0.628281, 0.555802, 0.366065, 1.0},
    51.2
  },
  { { 0.24725, 0.2245, 0.0645, 1.0},
    { 0.34615, 0.3143, 0.0903, 1.0},
    { 0.797357, 0.723991, 0.208006, 1.0},
    83.2
  },
  { { 0.105882, 0.058824, 0.113725, 1.0},
    { 0.427451, 0.470588, 0.541176, 1.0},
    { 0.333333, 0.333333, 0.521569, 1.0},
    9.84615 
  },
  { { 0.19225, 0.19225, 0.19225, 1.0},
    { 0.50754, 0.50754, 0.50754, 1.0},
    { 0.508273, 0.508273, 0.508273, 1.0},
    51.2
  },
  { { 0.23125, 0.23125, 0.23125, 1.0},
    { 0.2775, 0.2775, 0.2775, 1.0},
    { 0.773911, 0.773911, 0.773911, 1.0},
    89.6
  },
  { { 0.0215, 0.1745, 0.0215, 0.55},
    { 0.07568, 0.61424, 0.07568, 0.55},
    { 0.633, 0.727811, 0.633, 0.55},
    76.8
  },
  { { 0.135, 0.2225, 0.1575, 0.95},
    { 0.54, 0.89, 0.63, 0.95},
    { 0.316228, 0.316228, 0.316228, 0.95},
    12.8
  },
  { { 0.05375, 0.05, 0.06625, 0.82},
    { 0.18275, 0.17, 0.22525, 0.82},
    { 0.332741, 0.328634, 0.346435, 0.82},
    38.4
  },
  { { 0.25, 0.20725, 0.20725, 0.922},
    { 1.0, 0.829, 0.829, 0.922},
    { 0.296648, 0.296648, 0.296648, 0.922},
    11.264
  },
  { { 0.1745, 0.01175, 0.01175, 0.55},
    { 0.61424, 0.04136, 0.04136, 0.55},
    { 0.727811, 0.626959, 0.626959, 0.55},
    76.8
  },
  { { 0.1, 0.18725, 0.1745, 0.8},
    { 0.396, 0.74151, 0.69102, 0.8},
    { 0.297254, 0.30829, 0.306678, 0.8},
    12.8
  },
  { { 0.0, 0.0, 0.0, 1.0},
    { 0.01, 0.01, 0.01, 1.0},
    { 0.50, 0.50, 0.50, 1.0},
    32.0
  },
  { { 0.02, 0.02, 0.02, 1.0},
    { 0.01, 0.01, 0.01, 1.0},
    { 0.4, 0.4, 0.4, 1.0},
    10.0
  }
};

/* "zero" vector -- used for emissive property, since
** none of the materials described here are emissive
*/
GLfloat zero[4] = {0.0, 0.0, 0.0, 1.0};


/* the function to set the material properties */
void SetCurrentMaterial( int face, int num )
{
  glMaterialfv( (GLenum)face, GL_AMBIENT, mat[num].ambient );
  glMaterialfv( (GLenum)face, GL_DIFFUSE, mat[num].diffuse );
  glMaterialfv( (GLenum)face, GL_SPECULAR, mat[num].specular );
  glMaterialfv( (GLenum)face, GL_EMISSION, zero );
  glMaterialf( (GLenum)face, GL_SHININESS, mat[num].shiny );
}

/* the function to set the material properties */
void SetCurrentMaterialPlus( int face, int num, GLdouble amb[3], 
			     GLdouble dif[3], GLdouble spec[3] )
{
  GLfloat tmp[4];
  tmp[0] = mat[num].ambient[0] + amb[0];
  tmp[1] = mat[num].ambient[1] + amb[1];
  tmp[2] = mat[num].ambient[2] + amb[2];
  tmp[3] = mat[num].ambient[3];
  glMaterialfv( (GLenum)face, GL_AMBIENT, tmp );

  tmp[0] = mat[num].diffuse[0] + dif[0];
  tmp[1] = mat[num].diffuse[1] + dif[1];
  tmp[2] = mat[num].diffuse[2] + dif[2];
  tmp[3] = mat[num].diffuse[3];
  glMaterialfv( (GLenum)face, GL_DIFFUSE, tmp );

  tmp[0] = mat[num].specular[0] + spec[0];
  tmp[1] = mat[num].specular[1] + spec[1];
  tmp[2] = mat[num].specular[2] + spec[2];
  tmp[3] = mat[num].specular[3];
  glMaterialfv( (GLenum)face, GL_SPECULAR, tmp );
  glMaterialfv( (GLenum)face, GL_EMISSION, zero );
  glMaterialf( (GLenum)face, GL_SHININESS, mat[num].shiny );
}


void SetCurrentMaterialToColor( int face, float r, float g, float b )
{
  float amb = 0.25;
  float dif = 0.5;
  float spec = 0.7;
  float vect[4];
  
  /* not gonna be transparent */
  vect[3] = 1.0;

  /* set ambient color vector */
  vect[0] = r*amb; vect[1] = g*amb; vect[2] = b*amb;
  glMaterialfv( (GLenum)face, GL_AMBIENT, vect ); 

  /* set diffuse color vector */
  vect[0] = r*dif; vect[1] = g*dif; vect[2] = b*dif;
  glMaterialfv( (GLenum)face, GL_DIFFUSE, vect ); 

  /* set specular color vector */
  vect[0] = r*spec; vect[1] = g*spec; vect[2] = b*spec;
  glMaterialfv( (GLenum)face, GL_SPECULAR, vect ); 

  /* no emission */
  glMaterialfv( (GLenum)face, GL_EMISSION, zero );

  /* kinda shiny */
  glMaterialf( (GLenum)face, GL_SHININESS, 50.0 );

}

void SetCurrentMaterialToWhite( int face )
{
  GLfloat one[4] = {1.0, 1.0, 1.0, 1.0};
  
  /* set ambient color vector */
  glMaterialfv( (GLenum)face, GL_AMBIENT, one ); 

  /* set diffuse color vector */
  glMaterialfv( (GLenum)face, GL_DIFFUSE, one ); 

  /* set specular color vector */
  glMaterialfv( (GLenum)face, GL_SPECULAR, one ); 

  /* emission */
  glMaterialfv( (GLenum)face, GL_EMISSION, one );

  /* kinda shiny */
  glMaterialf( (GLenum)face, GL_SHININESS, 50.0 );

}
