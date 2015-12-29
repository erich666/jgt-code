
//--------------------------------------------------------------------------//
// lbvhTrace.cpp - A test ray tracer for lightweight bounding volumes 
//--------------------------------------------------------------------------//

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "Geometry.h"
#include "LBVH.h"

Camera theCamera;
Scene theScene;

bool parse(char *fileName);
bool parseCamera(FILE *F);
bool parseDirectionalLight(FILE *F);
bool parsePolyMesh(FILE *F, int objectsPerLeaf);
bool parsePly(TriangleMesh *mesh, char *meshName);

//--------------------------------------------------------------------------//
// MAIN FUNCTION
//--------------------------------------------------------------------------//

int main(int numArgs, char* args[])
{
	if (numArgs < 2) {
		printf("LBVHtrace inFile (imageName)\n");
		exit(0);
	}

	if (!parse(args[1])) { 
		printf("Unable to parse file '%s'", args[1]);
		exit(0);
	}

	if (numArgs > 2) {
		theCamera.captureImage(&theScene, args[2]);
	} else {
		theCamera.captureImage(&theScene, "image.ppm");
	}

	return 0;
}

//--------------------------------------------------------------------------//
// PARSER FUNCTIONS
//--------------------------------------------------------------------------//

bool stringStartsWith(char *s, char *val) 
{
	return !strncmp(s, val, strlen(val));
}

//--------------------------------------------------------------------------//

bool parse(char *fileName)
{
	double startTime = TIME();
	printf("PARSING '%s' ... ", fileName);

	int objectsPerLeaf = 1;
	char buff[256], dummy[256];
	FILE *F = fopen(fileName, "r");
	if (!F) return false;

	while (fgets(buff, 255, F)) {
		if (stringStartsWith(buff, "directionalLight")) {
			if (!parseDirectionalLight(F)) return false;
		} else if (stringStartsWith(buff, "polyMesh")) {
			if (!parsePolyMesh(F, objectsPerLeaf)) return false;
		} else if (stringStartsWith(buff, "backgroundColor")) {
			float r,g,b;
			sscanf(buff, "%s%f%f%f", dummy, &r, &g, &b);
			theScene.backgroundColor.set(r,g,b);
		} else if (stringStartsWith(buff, "ambientLight")) {
			float r,g,b;
			sscanf(buff, "%s%f%f%f", dummy, &r, &g, &b);
			theScene.ambientLight.set(r,g,b);
		} else if (stringStartsWith(buff, "objectsPerLeaf")) {
			sscanf(buff, "%s%d", dummy, &objectsPerLeaf);
			if (objectsPerLeaf < 1) objectsPerLeaf = 1;
		} else if (stringStartsWith(buff, "camera")) {
			if (!parseCamera(F)) return false;
		}
	}
	fclose(F);

	LBVH *rootNode = new LBVH();
	theScene.rootBoundingVolume = rootNode;
	rootNode->setBoundingVolumes(&theScene.meshes);
	rootNode->initHierarchy(1);

	double endTime = TIME();
	printf("done. (%0.3f seconds)\n", endTime-startTime);

	return true;
}

//--------------------------------------------------------------------------//

bool parseCamera(FILE *F)
{
	char buff[256], dummy[256];
	Point3 P;
	int w,h;
	float fov;

	while (fgets(buff, 255, F)) {
		if (stringStartsWith(buff, "lookFrom")) {
			sscanf(buff, "%s%f%f%f", dummy, &P[0], &P[1], &P[2]);
			theCamera.lookFrom = P;
		} else if (stringStartsWith(buff, "lookAt")) {
			sscanf(buff, "%s%f%f%f", dummy, &P[0], &P[1], &P[2]);
			theCamera.lookAt = P;
		} else if (stringStartsWith(buff, "viewUp")) {
			sscanf(buff, "%s%f%f%f", dummy, &P[0], &P[1], &P[2]);
			theCamera.viewUp = P;
		} else if (stringStartsWith(buff, "imageSize")) {
			sscanf(buff, "%s%d%d", dummy, &w, &h);
			theCamera.imageWidth = w;
			theCamera.imageHeight = h;
		} else if (stringStartsWith(buff, "fieldOfView")) {
			sscanf(buff, "%s%f", dummy, &fov);
			theCamera.fieldOfView = fov;
		} else if (stringStartsWith(buff, "end_camera")) {
			return true;
		}
	}
	return false;
}

//--------------------------------------------------------------------------//

bool parseDirectionalLight(FILE *F)
{
	char buff[256], dummy[256];
	Point3 color;
	Point3 direction;
	DirectionalLight *light = new DirectionalLight();

	while (fgets(buff, 255, F)) {
		if (stringStartsWith(buff, "color")) {
			sscanf(buff, "%s%f%f%f", dummy, &color[0], &color[1], &color[2]);
		} else if (stringStartsWith(buff, "direction")) {
			sscanf(buff, "%s%f%f%f", dummy, &direction[0], &direction[1], &direction[2]);
		} else if (stringStartsWith(buff, "end_directionalLight")) {
			light->set(direction, color);
			theScene.lights.push_back(light);
			return true;
		}
	}
	return false;
}

//--------------------------------------------------------------------------//

bool parsePolyMesh(FILE *F, int objectsPerLeaf)
{
	char buff[256], dummy[256], meshFile[256];
	Point3 scale, trans, diff, spec;
	TriangleMesh *triangleMesh = new TriangleMesh();
	LBVH *boundingVolume = new LBVH();
	boundingVolume->setTriangleMesh(triangleMesh);

	while (fgets(buff, 255, F)) {
		if (stringStartsWith(buff, "plyFile")) {
			sscanf(buff, "%s%s", dummy, meshFile);
			if (!parsePly(triangleMesh, meshFile)) return false;
		} else if (stringStartsWith(buff, "scale")) {
			sscanf(buff, "%s%f%f%f", dummy, &scale[0], &scale[1], &scale[2]);
			triangleMesh->scale(scale);
		} else if (stringStartsWith(buff, "translate")) {
			sscanf(buff, "%s%f%f%f", dummy, &trans[0], &trans[1], &trans[2]);
			triangleMesh->translate(trans);
		} else if (stringStartsWith(buff, "diffuse")) {
			sscanf(buff, "%s%f%f%f", dummy, &diff[0], &diff[1], &diff[2]);
			triangleMesh->material.diffuse = diff;
		} else if (stringStartsWith(buff, "specular")) {
			sscanf(buff, "%s%f%f%f", dummy, &spec[0], &spec[1], &spec[2]);
			triangleMesh->material.specular = spec;
		} else if (stringStartsWith(buff, "exponent")) {
			float exponent = 10.0f;
			sscanf(buff, "%s%f", dummy, &exponent);
			triangleMesh->material.exponent = exponent;
		} else if (stringStartsWith(buff, "end_polyMesh")) {
			boundingVolume->initHierarchy(objectsPerLeaf);
			theScene.meshes.push_back(boundingVolume);
			return true;
		}
	}

	return false;
}

//--------------------------------------------------------------------------//

int swapByteOrder(int A)
{
	int B = 0;
	char *a = (char*)&A;
	char *b = (char*)&B;
	b[0]=a[3]; b[1]=a[2]; b[2]=a[1]; b[3]=a[0];
	return B;
}

bool parsePly(TriangleMesh *mesh, char *meshFile)
{
	int i,j;
	int numVertices = 0;
	int numFaces = 0;
	bool hasNormals = false;
	char buff[256];
	char *token;
	Point3 V, N;
	Triangle T;
	int verts, v0,v1,v2;

	FILE *F = fopen(meshFile, "r+b");
	if (!F) return false;

	// read the file header
	while (fgets(buff, 255, F)) {
		if (stringStartsWith(buff, "end_header")) {
			break;
		} else if (stringStartsWith(buff, "element vertex")) {
			sscanf(&buff[strlen("element vertex")], "%d", &numVertices);
		} else if (stringStartsWith(buff, "property float nx")) {
			hasNormals = true;
		} else if (stringStartsWith(buff, "element face")) {
			sscanf(&buff[strlen("element face")], "%d", &numFaces);
		}
	}
	mesh->triangles.reserve(numFaces);
	mesh->vertices.reserve(numVertices);
	if (hasNormals) mesh->normals.reserve(numVertices);

	// read vertices, normals
	for (i=0; i<numVertices; i++) {
		fgets(buff, 255, F);
		if (hasNormals) {
			sscanf(buff, "%f%f%f%f%f%f", &V[0], &V[1], &V[2], &N[0], &N[1], &N[2]);
			mesh->vertices.push_back(V);
			mesh->normals.push_back(N);
		} else {
			sscanf(buff, "%f%f%f", &V[0], &V[1], &V[2]);
			mesh->vertices.push_back(V);
		}
	}
	// read faces
	for (i=0; i<numFaces; i++) {
		fgets(buff, 255, F);
		token = strtok(buff, " ");
		verts = atoi(token);
		token = strtok(NULL, " ");
		v0 = atoi(token);
		token = strtok(NULL, " ");
		v1 = atoi(token);
		for (j=2; j<verts; j++) {
			token = strtok(NULL, " ");
			v2 = atoi(token);
			T.setVertices(v0, v1, v2);
			mesh->triangles.push_back(T);
			v1 = v2;
		}
	}

	fclose(F);
	return true;
}
//--------------------------------------------------------------------------//
