#ifndef __TSPLAT_H
#define __TSPLAT_H

#include <stddef.h>
#include <GL/glut.h>
#include "datatype.h"

enum SPLAT_TYPE {
	CIRCLE,
	HEXAGON
};

class TextureSplat {
public:
	TextureSplat();
	~TextureSplat();

	int Width()  { return width;  };
	int Height() { return height; };
	virtual void InitRender()=0;
	virtual void InitRenderNV()=0;	
	virtual void RenderList()=0;
	virtual void Render(GLfloat intens)=0;
	virtual void Render(GLfloat r, GLfloat g, GLfloat b, GLfloat a)=0;
	virtual void RenderNV(GLfloat intens)=0;
	virtual void RenderNV(GLfloat r, GLfloat g, GLfloat b, GLfloat a)=0;

public:
	int width, height;
	GLuint texName;
	GLuint splatList;
};

class GaussTextureSplat : public TextureSplat {
public:
	GaussTextureSplat() {};
	void InitSplat(int w, int h, SCALAR s, SCALAR kernel_radius, SPLAT_TYPE type);
	void SetTransferColor(SCALAR r, SCALAR g, SCALAR b) {red=r;green=g;blue=b;};
	virtual void InitRender();
	virtual void InitRenderNV();
	virtual void RenderList();
	virtual void Render(GLfloat intens);
	virtual void Render(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
	virtual void RenderNV(GLfloat intens);
	virtual void RenderNV(GLfloat r, GLfloat g, GLfloat b, GLfloat a);

private:
	void InitSplatTexture(GLfloat *image);

	SPLAT_TYPE s_type;
	SCALAR sigma;
	double radius;
	SCALAR red, green, blue;
};

#endif /* __TSPLAT_H */
