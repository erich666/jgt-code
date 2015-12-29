/*************************************************
** readDotM.cpp                                 **
** -----------                                  **
**                                              **
** Basic code for reading a .m model file.      **
**                                              **
** Chris Wyman (9/07/2006)                      **
*************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "DotM.h"

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif 

#ifndef MAX
#define MAX(x,y)   ((x)>(y)?(x):(y))
#endif


#include "glut_template.h"

void FileFirstPass( FILE *f, mymodel *m )
{
  int vcount = 0;
  int tcount = 0;
  char buf[2048];
  
  while( !feof(f) )
    {
	  buf[0] = 0;
      fgets( buf, 2048, f );
      switch( buf[0] )
	{
	  default:
		  break;
	case 'V':
	case 'v':
	  vcount++;
	  break;
	case 'F':
	case 'f':
	  tcount++;
	  break;
	}
    }
  m->numVertices = vcount;
  m->numTriangles = tcount;
}

void FileSecondPass( FILE *f, mymodel *m )
{
  char buf[2048];
  int num, ret, idx[3], more;
  double pos[3], norm[3], sphr[3], uv[2];
  
  m->sphereMapped = 1;
  m->uvMapped = 1;
  m->vertexNormals = 1;

  while( !feof(f) )
    {
      fgets( buf, 2048, f );
      switch( buf[0] )
	{
	  /* we found a vertex!! */
	case 'V':
	case 'v':
	  /* read in the vertex */
	  ret = sscanf( buf, "%*s %d %lf %lf %lf {normal=(%lf %lf %lf) sph=(%lf %lf %lf) uv=(%lf %lf)}",
			&num, 
			&pos[0], &pos[1], &pos[2], 
			&norm[0], &norm[1], &norm[2], 
			&sphr[0], &sphr[1], &sphr[2],
			&uv[0], &uv[1] );

	  /* we'd better have read at least 4 entries on this line! */
	  assert( ret >= 4 );
	    
	  /* don't want to try to index a vertex that doesn't exist! */
	  assert( num <= m->numVertices );

	  m->vertex[num-1].pos[0] = pos[0];
	  m->vertex[num-1].pos[1] = pos[1];
	  m->vertex[num-1].pos[2] = pos[2];

	  /* check to see what other data we read */
	  if (ret < 7)
	    m->vertexNormals = 0;
	  else 
	    {
	      /* we got some vertex normals! */
	       m->vertex[num-1].norm[0] = norm[0];
	       m->vertex[num-1].norm[1] = norm[1];
	       m->vertex[num-1].norm[2] = norm[2];
	    }
	  if (ret < 10)
	    m->sphereMapped = 0;
	  else
	    {
	      /* we got a spherical parameterization! */
	      m->vertex[num-1].sphr[0] = sphr[0];
	      m->vertex[num-1].sphr[1] = sphr[1];
	      m->vertex[num-1].sphr[2] = sphr[2];
	    }

	  if (ret < 12)
	  {
	    m->uvMapped = 0;
		printf("vertex %d not uv-mapped!\n", num);
	  }
	  else
	    {
	      /* we got a uv mapping! */
	      m->vertex[num-1].uv[0] = uv[0];
	      m->vertex[num-1].uv[1] = uv[1];
	    }

	  break;

	  /* we found a face!! */
	case 'F':
	case 'f':
	  /* read in the vertex */
	  ret = sscanf( buf, "%*s %d %d %d %d",
			&num, 
			&idx[0], &idx[1], &idx[2], &more );

	  /* we'd better have read at least 4 entries on this line! */
	  assert( ret >= 4 );
	    
	  /* don't want to try to index a vertex that doesn't exist! */
	  assert( num <= m->numTriangles );

	  m->tri[num-1].vertex[0] = idx[0] - 1;
	  m->tri[num-1].vertex[1] = idx[1] - 1;
	  m->tri[num-1].vertex[2] = idx[2] - 1;

	  if (ret > 4)
	    printf("Warning: Detected faces with more than 3 vertices!  Code currently only handles triangles!\n");

	  break;
	}
    }

}



mymodel* ReadDotM(char* filename)
{
  mymodel* model;
  FILE*     file;

  /* open the file */
  file = fopen(filename, "r");
  assert( file );

  /* allocate a new model */
  model = (mymodel*)malloc(sizeof(mymodel));  
  assert(model);

  /* make sure we initialize it as expected */
  model->pathname      = strdup(filename);
  model->numVertices   = 0;
  model->vertex        = 0;
  model->numTriangles  = 0;
  model->tri           = 0;
  model->sphereMapped  = 0;
  model->uvMapped      = 0;
  model->faceNormals   = 0;
  model->vertexNormals = 0;
  model->number        = -1;

  /* read how many triangles & vertices are in the file */
  FileFirstPass( file, model );
  rewind( file );

  /* allocate memory */
  model->tri    = (mytriangle*)malloc( model->numTriangles * sizeof( mytriangle ) );
  model->vertex = (myvertex*)malloc( model->numVertices * sizeof( myvertex ) );
  assert( model->tri );
  assert( model->vertex );
  
  /* actually read the data into memory */
  FileSecondPass( file, model );

  /* close the file and return */
  fclose(file);

  //printf("(+) Finished reading model '%s'\n", filename );
  return model;
}


void FreeDotM( mymodel* m )
{
  if (!m) return;
  if (m->vertex) free( m->vertex );
  if (m->tri) free( m->tri );
  if (m->pathname) free( m->pathname );
  free( m );
}


double *ConvertSphrCoord2UV( double *uv, double *sphr)
{
	uv[1] = acos( sphr[2] )/M_PI;
	uv[0] = atan2( sphr[1], sphr[0] )/(2*M_PI);
	if (uv[0] < 0) uv[0]+=1;
	return uv;
}


void DrawDotM( mymodel* m, double *nDist, float* curvatureData )
{
  int i;
  double tmp[2];

  /* make sure we've got a model to draw */
  assert(m);
  if (m->numVertices <= 0 || m->numTriangles <= 0) return;
  assert(m->vertex);
  assert(m->tri);

  glBegin(GL_TRIANGLES);
  for (i = 0; i < m->numTriangles; i++) 
    {
	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[0] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[0] + 1],
											curvatureData[ 4 * m->tri[i].vertex[0] + 2],
											curvatureData[ 4 * m->tri[i].vertex[0] + 3] );
      if (m->uvMapped)
		glTexCoord2dv( m->vertex[m->tri[i].vertex[0]].uv );
      if (m->vertexNormals)
		//glNormal3dv( m->vertex[m->tri[i].vertex[0]].norm );
	    glVertexAttrib4dARB( 2, m->vertex[m->tri[i].vertex[0]].norm[0],
								m->vertex[m->tri[i].vertex[0]].norm[1], 
								m->vertex[m->tri[i].vertex[0]].norm[2],
								nDist[m->tri[i].vertex[0]]  );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[0]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[0]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[0]].pos );
      

	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[1] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[1] + 1],
											curvatureData[ 4 * m->tri[i].vertex[1] + 2],
											curvatureData[ 4 * m->tri[i].vertex[1] + 3] );
      if (m->uvMapped)
		glTexCoord2dv( m->vertex[m->tri[i].vertex[1]].uv );
      if (m->vertexNormals)
		//glNormal3dv( m->vertex[m->tri[i].vertex[1]].norm );
		glVertexAttrib4dARB( 2, m->vertex[m->tri[i].vertex[1]].norm[0],
								m->vertex[m->tri[i].vertex[1]].norm[1], 
								m->vertex[m->tri[i].vertex[1]].norm[2],
								nDist[m->tri[i].vertex[1]]  );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[1]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[1]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[1]].pos );
      

	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[2] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[2] + 1],
											curvatureData[ 4 * m->tri[i].vertex[2] + 2],
											curvatureData[ 4 * m->tri[i].vertex[2] + 3] );
      if (m->uvMapped)
		glTexCoord2dv( m->vertex[m->tri[i].vertex[2]].uv );
      if (m->vertexNormals)
		//glNormal3dv( m->vertex[m->tri[i].vertex[2]].norm );
		glVertexAttrib4dARB( 2, m->vertex[m->tri[i].vertex[2]].norm[0],
								m->vertex[m->tri[i].vertex[2]].norm[1], 
								m->vertex[m->tri[i].vertex[2]].norm[2],
								nDist[m->tri[i].vertex[2]]  );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[2]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[2]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[2]].pos );
          
    }
  glEnd();
}


double Dot( double *a, double *b )
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void Mul( double *a, double f, double *b )
{
	a[0] = f*b[0];
	a[1] = f*b[1];
	a[2] = f*b[2];
}

void Sub( double *a, double *b, double *c )
{
	a[0] = b[0]-c[0];
	a[1] = b[1]-c[1];
	a[2] = b[2]-c[2];
}

void MapSphericalParamToOctahedronDotM( mymodel* m )
{
	double x[2][3] = { {1,0,0}, {-1,0,0} };
	double y[2][3] = { {0,1,0}, {0,-1,0} };
	double z[2][3] = { {0,0,1}, {0,0,-1} };
	double uvX[2][2] = { { 0.5, 0.5 }, {0, 0} };
	double uvY[2][2] = { { 1, 0.5}, {0, 0.5} };
	double uvZ[2][2] = { { 0.5, 1}, {0.5, 0} };
	double sqrt1_2 = sqrt(2.0)/2.0;
	double norm[8][3] = 
		{ {sqrt(3.0), sqrt(3.0), sqrt(3.0)},
		  {sqrt(3.0), sqrt(3.0), -sqrt(3.0)},
		  {sqrt(3.0), -sqrt(3.0), sqrt(3.0)},
		  {sqrt(3.0), -sqrt(3.0), -sqrt(3.0)},
		  {-sqrt(3.0), sqrt(3.0), sqrt(3.0)},
		  {-sqrt(3.0), sqrt(3.0), -sqrt(3.0)},
		  {-sqrt(3.0), -sqrt(3.0), sqrt(3.0)},
		  {-sqrt(3.0), -sqrt(3.0), -sqrt(3.0)} };
	double *mapPos, planePos[3], tmp[3];
	double cosNorm;
	int ct, i;

	if (m->sphereMapped) return;

	assert( m->sphereMapped );
	for (i=0; i < m->numVertices; i++)
	{
		mapPos = m->vertex[i].sphr;

		/* find which face of the octahedron this is in */
		for	(ct = 0; ct < 7; ct++ )
		{
			if ( Dot( mapPos, x[ (ct&0x4) ] ) < 0 ) continue;
			if ( Dot( mapPos, y[ (ct&0x2) ] ) < 0 ) continue;
			if ( Dot( mapPos, z[ (ct&0x1) ] ) < 0 ) continue;
			break;
		}	
		assert( ct < 8 );

		/* Find the non-normal component of the vector (i.e. on the face) */
		cosNorm = Dot( mapPos, norm[ct] );
		Mul( tmp, cosNorm, mapPos );
		Sub( planePos, mapPos, tmp );
		Mul( planePos, sqrt1_2/cosNorm, planePos );
		/* could find location on the face here by adding sqrt1_2*norm[ct] to planePos */

		

	}

}


void UnitizeDotM( mymodel* m )
{
	int i;
	double max[3], min[3];
	double d[3], delta;
	assert(m);
	
	max[0] = min[0] = m->vertex[0].pos[0];
	max[1] = min[1] = m->vertex[0].pos[1];
	max[2] = min[2] = m->vertex[0].pos[2];
	for( i = 0; i < m->numVertices; i++ )
	{
		if (m->vertex[i].pos[0] > max[0]) max[0] = m->vertex[i].pos[0];
		if (m->vertex[i].pos[0] < min[0]) min[0] = m->vertex[i].pos[0];
		if (m->vertex[i].pos[1] > max[1]) max[1] = m->vertex[i].pos[1];
		if (m->vertex[i].pos[1] < min[1]) min[1] = m->vertex[i].pos[1];
		if (m->vertex[i].pos[2] > max[2]) max[2] = m->vertex[i].pos[2];
		if (m->vertex[i].pos[2] < min[2]) min[2] = m->vertex[i].pos[2];
	}

	d[0] = max[0]+min[0];
	d[1] = max[1]+min[1];
	d[2] = max[2]+min[2];

	delta = MAX( (max[0]-min[0]), MAX( (max[1]-min[1]), (max[2]-min[2]) ) );
	delta = 2/delta;

	for ( i = 0; i < m->numVertices; i++ )
	{
		m->vertex[i].pos[0] = (m->vertex[i].pos[0] - (d[0]/2.0)) * delta;
		m->vertex[i].pos[1] = (m->vertex[i].pos[1] - (d[1]/2.0)) * delta;
		m->vertex[i].pos[2] = (m->vertex[i].pos[2] - (d[2]/2.0)) * delta;
	}
}


void DrawDotMNormals( mymodel* m, double *nDist, float* curvatureData )
{
  int i;
  double tmp[2];

  /* make sure we've got a model to draw */
  assert(m);
  if (m->numVertices <= 0 || m->numTriangles <= 0) return;
  assert(m->vertex);
  assert(m->tri);
  assert(m->sphereMapped);

  glBegin(GL_TRIANGLES);
  for (i = 0; i < m->numTriangles; i++) 
    {
	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[0] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[0] + 1],
											curvatureData[ 4 * m->tri[i].vertex[0] + 2],
											curvatureData[ 4 * m->tri[i].vertex[0] + 3] );
	  glColor4d( (m->vertex[m->tri[i].vertex[0]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[0]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[0]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[0]] / (2*sqrt(3.0)) );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[0]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[0]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[0]].pos );

		
	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[1] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[1] + 1],
											curvatureData[ 4 * m->tri[i].vertex[1] + 2],
											curvatureData[ 4 * m->tri[i].vertex[1] + 3] );
	  glColor4d( (m->vertex[m->tri[i].vertex[1]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[1]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[1]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[1]] / (2*sqrt(3.0)) );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[1]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[1]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[1]].pos );
      

	  if (curvatureData)
			glMultiTexCoord4f( GL_TEXTURE7, curvatureData[ 4 * m->tri[i].vertex[2] + 0],
				                            curvatureData[ 4 * m->tri[i].vertex[2] + 1],
											curvatureData[ 4 * m->tri[i].vertex[2] + 2],
											curvatureData[ 4 * m->tri[i].vertex[2] + 3] );
	  glColor4d( (m->vertex[m->tri[i].vertex[2]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[2]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[2]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[2]] / (2*sqrt(3.0)) );
	  if (m->sphereMapped)
	  {
		glMultiTexCoord2dv( GL_TEXTURE1, ConvertSphrCoord2UV( tmp, m->vertex[m->tri[i].vertex[2]].sphr ) );
		glMultiTexCoord3dv( GL_TEXTURE2, m->vertex[m->tri[i].vertex[2]].sphr );
	  }
      glVertex3dv( m->vertex[m->tri[i].vertex[2]].pos );
          
    }
  glEnd();
}


void DrawPlanarDotMNormals( mymodel* m, double *nDist )
{
  int i;

  /* make sure we've got a model to draw */
  assert(m);
  if (m->numVertices <= 0 || m->numTriangles <= 0) return;
  if (!m->uvMapped) return;
  assert(m->vertex);
  assert(m->tri);
  assert(m->uvMapped);

  glBegin(GL_TRIANGLES);
  for (i = 0; i < m->numTriangles; i++) 
    {
	  if ( fabs(m->vertex[m->tri[i].vertex[0]].uv[0] - m->vertex[m->tri[i].vertex[1]].uv[0]) > 0.2 ) continue;
	  if ( fabs(m->vertex[m->tri[i].vertex[0]].uv[0] - m->vertex[m->tri[i].vertex[2]].uv[0]) > 0.2 ) continue;
	  if ( fabs(m->vertex[m->tri[i].vertex[0]].uv[1] - m->vertex[m->tri[i].vertex[1]].uv[1]) > 0.2 ) continue;
	  if ( fabs(m->vertex[m->tri[i].vertex[0]].uv[1] - m->vertex[m->tri[i].vertex[2]].uv[1]) > 0.2 ) continue;

	  glColor4d( (m->vertex[m->tri[i].vertex[0]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[0]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[0]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[0]] / (2*sqrt(3.0)) );
      glVertex2dv( m->vertex[m->tri[i].vertex[0]].uv );

	  glColor4d( (m->vertex[m->tri[i].vertex[1]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[1]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[1]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[1]] / (2*sqrt(3.0)) );
      glVertex2dv( m->vertex[m->tri[i].vertex[1]].uv );
      
	  glColor4d( (m->vertex[m->tri[i].vertex[2]].norm[0]+1)/2.0,
				  (m->vertex[m->tri[i].vertex[2]].norm[1]+1)/2.0, 
				  (m->vertex[m->tri[i].vertex[2]].norm[2]+1)/2.0,
				  nDist[m->tri[i].vertex[2]] / (2*sqrt(3.0)) );
      glVertex2dv( m->vertex[m->tri[i].vertex[2]].uv );
          
    }
  glEnd();
}

GLuint CreateDotMList( mymodel* model, double *nDist, float* curvatureData )
{
  GLuint list;
  list = glGenLists(1);
  glNewList(list, GL_COMPILE);
  DrawDotM(model, nDist, curvatureData);
  glEndList();

  return list;
}

GLfloat GetDotMBoundingSphereRadius( mymodel* model, GLfloat *center )
{
  GLint  i;
  GLfloat maxRadius = 0;
  GLfloat cx, cy, cz, maxx, minx, maxy, miny, maxz, minz;
  GLfloat tmpR;

  assert(model);
  assert(model->vertex);

  /* get the max/mins */
  maxx = minx = model->vertex[1].pos[0];
  maxy = miny = model->vertex[1].pos[1];
  maxz = minz = model->vertex[1].pos[2];
  for (i = 1; i < model->numVertices; i++) 
  {
    if (maxx < model->vertex[i].pos[0])
      maxx = model->vertex[i].pos[0];
    if (minx > model->vertex[i].pos[0])
      minx = model->vertex[i].pos[0];

    if (maxy < model->vertex[i].pos[1])
      maxy = model->vertex[i].pos[1];
    if (miny > model->vertex[i].pos[1])
      miny = model->vertex[i].pos[1];

    if (maxz < model->vertex[i].pos[2])
      maxz = model->vertex[i].pos[2];
    if (minz > model->vertex[i].pos[2])
      minz = model->vertex[i].pos[2];
  }

  /* calculate center of the model */
  center[0] = cx = (maxx + minx) / 2.0;
  center[1] = cy = (maxy + miny) / 2.0;
  center[2] = cz = (maxz + minz) / 2.0;

  /* find the actual radius of the bounding sphere... */
  for (i = 1; i <= model->numVertices; i++) 
  {
	  tmpR = (model->vertex[i].pos[0]-center[0]) * (model->vertex[i].pos[0]-center[0]) +
			(model->vertex[i].pos[1]-center[1]) * (model->vertex[i].pos[1]-center[1]) +
			(model->vertex[i].pos[2]-center[2]) * (model->vertex[i].pos[2]-center[2]);

	  tmpR = sqrt( tmpR );
	  if (tmpR > maxRadius) maxRadius = tmpR;
  }

  return maxRadius;
}