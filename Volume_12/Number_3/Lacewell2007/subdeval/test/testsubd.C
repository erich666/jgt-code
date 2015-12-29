/*
Brent Burley, Oct. 2006
This test program reads a mesh, samples the limit surface at a specified 
number of UV locations, then saves (p, dPdU, dPdV) to an output file.  At 
each limit point (p) vectors dPdU and dPdV are saved as short line 
segments.
*/

#include <assert.h>
#include <sys/time.h>
#include <vector>
#include "Subd.h"

class timer
{
 public:
    timer() { gettimeofday(&t1, 0); }
    void stop() { gettimeofday(&t2, 0);
	printf("%g\n", (t2.tv_usec-t1.tv_usec)*1e-6+t2.tv_sec-t1.tv_sec);
    }
    timeval t1, t2;
};

bool loadOBJ(const char* filename, 
	     std::vector<float>& verts,
	     std::vector<int>& nvertsPerFace,
	     std::vector<int>& faceverts)
{
    FILE* file = fopen(filename, "r");
    if (!file)
    {
	printf("Failed to open file %s.\n", filename);
	return false;
    }
    
    char line[256];
    while (fgets(line, sizeof(line), file))
    {
	char* end = &line[strlen(line)-1];
	if (*end == '\n') *end = '\0'; // strip trailing nl
	float x, y, z, u, v;
	switch (line[0]) {
	case 'v':
	    switch (line[1]) {
	    case ' ':
		if (sscanf(line, "v %f %f %f", &x, &y, &z) == 3) {
		    verts.push_back(x);
		    verts.push_back(y);
		    verts.push_back(z);
		}
		break;
	    case 't':
		if (sscanf(line, "vt %f %f", &u, &v) == 2) {
		    // ignore assigned UVs
		}
		break;
	    }
	    break;
	case 'f':
	    if (line[1] == ' ') {
		int vi, ti, ni;
		const char* cp = &line[2];
		while (*cp == ' ') cp++;
		int nverts = 0;
		while (sscanf(cp, "%d/%d/%d", &vi, &ti, &ni) == 3) {
		    nverts++;
		    faceverts.push_back(vi-1);
		    while (*cp && *cp != ' ') cp++;
		    while (*cp == ' ') cp++;
		}
		nvertsPerFace.push_back(nverts);
	    }
	    break;
	}
    }
    printf("read %d faces\n", nvertsPerFace.size());
    fclose(file);
    return true;
}


bool saveObj(const char* filename, Subd* subd, bool limit)
{
    FILE* file = fopen(filename, "w");
    if (!file)
    {
	printf("Failed to open file %s for write.\n", filename);
	return false;
    }
    
    fprintf(file, "g default\n");
    const float* v = limit ? subd->limitverts() : subd->verts();
    for (int i = 0; i < subd->nverts(); i++, v += 3) {
	fprintf(file, "v %g %g %g\n", v[0], v[1], v[2]);
    }
    
    const float* n = subd->normals();
    for (int i = 0; i < subd->nverts(); i++, n += 3) {
	fprintf(file, "vn %g %g %g\n", n[0], n[1], n[2]);
    }

    fprintf(file, "g subd\n");
    const int* nv = subd->nvertsPerFace();
    const int* fv = subd->faceverts();
    printf("%d\n", subd->nfaces());
    for (int i = 0; i < subd->nfaces(); i++, nv++) {
	fprintf(file, "f");
	for (int j = 0; j < *nv; j++, fv++) {
	    fprintf(file, " %d//%d", *fv+1, *fv+1);
	}
	fprintf(file, "\n");
    }

    return true;
}


int vertid=1;
FILE* fp = 0;

void saveObj(const char* filename)
{
    fp = fopen(filename, "w");
    fprintf(fp, "g default\n");
}

void addLine(double* p, double* d)
{
    fprintf(fp, "v %g %g %g\n", p[0], p[1], p[2]);
    d[0] *= .05;
    d[1] *= .05;
    d[2] *= .05;
    fprintf(fp, "v %g %g %g\n", p[0]+d[0], p[1]+d[1], p[2]+d[2]);
    fprintf(fp, "v %g %g %g\n", p[0]+d[0], p[1]+d[1], p[2]+d[2]);
    fprintf(fp, "v %g %g %g\n", p[0], p[1], p[2]);
    fprintf(fp, "f %d %d %d %d\n", vertid, vertid+1, vertid+2, vertid+3);
    vertid += 4;
}


void printUsage(const char * progname)
{
    printf("usage: %s input.obj output.obj numsamples\n", progname);
    printf("input mesh is sampled, with (p, dPdU, dPdV) samples written to output file\n");
}


int main(int argc, char** argv)
{
    if (argc != 4) {
	printUsage(argv[0]);
	return 1;
    }
    
    std::vector<float> verts;
    std::vector<int> nvertsPerFace;
    std::vector<int> faceverts;
    if (!loadOBJ(argv[1], verts, nvertsPerFace, faceverts)) {
	printf("could not load %s\n", argv[1]);
	return 1;
    }

    saveObj(argv[2]);
    for (int i = 0; i < 1; i++) {
	Subd subd(verts.size()/3, &verts[0], nvertsPerFace.size(), 
		&nvertsPerFace[0], &faceverts[0]);
	//	timer t;
	// 	for (int i = 0; i < 1; i++) {
	// 	    subd.subdivide(2);
	// 	}
	//	t.stop();
	//	saveObj(argv[2], &subd, /*limit=*/ 1);
	int numsamples = atoi(argv[3]);
	double du = 1.0 / (numsamples-1);
	double dv = du;
	for (int f = 0; f < subd.nfaces(); f++) {
	    double u = 0.0;
	    for (int i = 0; i < numsamples; u += du, i++) {
		double v = 0.0;
		for (int j = 0; j < numsamples; v += dv, j++) {
		    double p[3], dpdu[3], dpdv[3];
		    subd.eval(f, u, v, p, dpdu, dpdv);
		    addLine(p, dpdu);
		    addLine(p, dpdv);
		}
	    }
	}
    }
    return 0;
}

