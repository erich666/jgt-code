/******************************************
 ** DotM.h                               **
 ** ------                               **
 **                                      **
 ** Code for reading in model files like **
 **    'bunny.m'  Note that this code    **
 **    also reads in per-vertex normals, **
 **    spherical parameterizations,      **
 **    and uv-coordinates like the test  **
 **    files give on Hugues Hoppe's page **
 **    from his & Emil Praun's "Sphrical **
 **    Parameterization and Remeshing"   **
 **    Siggraph 2003 paper.              **
 **                                      **
 ** Chris Wyman (11/03/2004)             **
 *****************************************/    
 
#ifndef DOTM_H
#define DOTM_H


// should probably split out properties other than pos
//    our of here (to save memory when non-existant)
typedef struct _myvertex {
  double pos[3];
  double norm[3];
  double sphr[3];
  double uv[2];
} myvertex;

// should probably split out properties other than vertex
//    indices out of here (to save memory when non-existant)
typedef struct _mytriangle {
  int vertex[3];
  double norm[3];
} mytriangle;

typedef struct _mymodel {
  int number;
  char *pathname;      // Where did we load this file from?

  int numVertices;     // How many vertices do we have?
  int numTriangles;    // How many faces do we have?

  int uvMapped;        // Do we have uv mappings for vertices?
  int sphereMapped;    // Do we have sphere mappings for vertices?
  int vertexNormals;   // Do we have vertex normals?
  int faceNormals;     // Do we have face normals?

  myvertex   *vertex;
  mytriangle *tri; 

} mymodel;














mymodel* ReadDotM(char* filename);
void FreeDotM( mymodel* m );
void DrawDotM( mymodel* m, double *nDist, float* curvatureData );
void UnitizeDotM( mymodel* m );
GLuint CreateDotMList( mymodel* model, double *nDist, float* curvatureData );
void DrawDotMNormals( mymodel* m, double *nDist, float* curvatureData );
void DrawPlanarDotMNormals( mymodel* m, double *nDist );
GLfloat GetDotMBoundingSphereRadius( mymodel* model, GLfloat *center );

#endif
