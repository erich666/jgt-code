#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <glut.h>
#include "field.h"
#include "flags.h"
#include "image.h"
#include "genrl.h"
#include "io.h"
#include "mfmm.h"



//----------------------------------------------------------------


FIELD<float>* compute_distance(FIELD<float>*,float,float);
void compute_gradient(FIELD<float>*,FIELD<float>*& gx,FIELD<float>*& gy);
void mask_image(IMAGE<float>*,FLAGS* m);
void display();
void inpaint();
void reshape(int,int);
float length;
int wnd1,wnd2;
FIELD<float>* grad_x,*grad_y;				//the complete (inside/outside) distance-gradient field
FIELD<float>* dist;					//the complete (inside/outside) distance-to-bounday field
IMAGE<float>* rgb_image;				//the full RGB image we inpaint
FIELD<float>* f;
FLAGS* flags;						//the FMM flags-field
int   B_radius = 5;					//the inpainting neighborhood radius
int   dst_wt = 1;					//use dist-weighting in inpainting (t/f)
int   lev_wt = 1;					//use level-weighting in inpainting (t/f)
const int   N = 0;					//window for smoothing during gradient computation
							//(if N=0, no smoothing is attempted)

//---------------------------------------------------------------------------------------------------
const float S = 0.8, G = 0.2;
const float S8 = S/8;
const float SG = 3*S/4 - G;
const float S4 = -4*S;

float w[5][5] = {{S8, 0, SG, 0, S8},{0, 0, S4, 0, 0},
{SG, S4, 1+4*G+12.5*S, S4, SG},{0, 0, S4, 0, 0},{S8, 0, SG, 0, S8}};

int main(int argc,char* argv[])
{
   glutInit(&argc, argv);

   argc--;argv++;					//Skip program name arg
   char inp[100],img[100];
   float k = -1;					//Threshold
   float sk_lev = 20;					//
   int  twopass = 1;					//Using 2-pass or 1-pass method for boundary treatment 
  
   if (argc<1)						//No cmdline args, read them interactively
   {
      cout<<"Original image: "; cin>>img; 
      cout<<"Scratch image:  "; cin>>inp;
   }
   else if (argc<2)                                     //One arg given, namely the input-image
   {  
      strcpy(inp,argv[0]); argc--;argv++; 
      cout<<"Scratch image:  "; cin>>inp;
   }   
   else
   {
      strcpy(inp,argv[0]); argc--;argv++; 		//arg1: name of scalar field
      strcpy(img,argv[0]); argc--;argv++;
      for(;argc;)					//Process cmdline args
      {
	char* opt = argv[0]+1; argc--;argv++;
        char* val = argv[0];

	if (opt[0]=='t') 				// -t <threshold>
	{  k = atof(val); argc--;argv++;  }
      }
   }
   if (!inp[0]) sprintf(inp,"scratch_%s",img);		//if no scratch image given, infer its name from original image


   f = FIELD<float>::read(inp);	                        //read scratch image	
   if (!f) { cout<<"Can not open file: "<<inp<<endl; return 1; }
   rgb_image = IMAGE<float>::read(img);
   if (!rgb_image) { cout<<"Can not open file: "<<img<<endl; return 1; }	  

   dist    = compute_distance(f,k,2*B_radius);          //compute complete distance field in a band 2*B_radius around the inpainting zone
   compute_gradient(dist,grad_x,grad_y);                //compute smooth gradient of distance field

   flags = new FLAGS(*f,k);
   mask_image(rgb_image,flags);                         //mask image with 'f' to produce defects
   inpaint();						//do the inpainting

   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   glutInitWindowSize(f->dimX(),f->dimY());
   wnd1 = glutCreateWindow("Window 1");
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutMainLoop();	

   delete f;
   delete flags;
   return 0;
}



void inpaint()
{
   int nfail,nextr;
   FLAGS* fl = new FLAGS(*flags); FIELD<float>* ff = new FIELD<float>(*f);
   ModifiedFastMarchingMethod mfmm(ff,fl,rgb_image,grad_x,grad_y,dist,B_radius,dst_wt,lev_wt,1000000);
   mfmm.execute(nfail,nextr); delete fl; delete ff; 
   rgb_image->normalize();
}


FIELD<float>* compute_distance(FIELD<float>* fi,float k,float maxd)
{
   int nfail,nextr;
   FIELD<float>*    fin = new FIELD<float>(*fi);	//Copy input field 
   FLAGS*   	flagsin = new FLAGS(*fin,k);		//Make flags field
   FLAGS*       fcopy   = new FLAGS(*flagsin);          //Copy flags field for combining the two fields afterwards
   FastMarchingMethod fmmi(fin,flagsin);
   fmmi.execute(nfail,nextr);

   FIELD<float>*   fout = new FIELD<float>(*fi);	//Copy input field 
   FLAGS*      flagsout = new FLAGS(*fout,-k);		//Make flags field    
   FastMarchingMethod fmmo(fout,flagsout);
   fmmo.execute(nfail,nextr,2*B_radius);		//Executr FMM only in a band 2*B_radius deep, we need no more

   FIELD<float>* f = new FIELD<float>(*fin);		//Combine in and out-fields in a single distance field 'f'
   for(int i=0;i<f->dimX();i++)			
     for(int j=0;j<f->dimY();j++)
     {
        if (fcopy->alive(i,j)) f->value(i,j) = -fout->value(i,j);
	if (flagsout->faraway(i,j)) f->value(i,j) = 0;
     }
   
   delete flagsin; delete flagsout; delete fin; delete fout; delete fcopy;
   return f;						//All done, return 'f'
}


void mask_image(IMAGE<float>* f,FLAGS* m)			//here we throw away the info to inpaint...
{				
   for(int i=0;i<f->dimX();i++)
     for(int j=0;j<f->dimY();j++)
        if (m->faraway(i,j)) f->setValue(i,j,0);
}


void gradient_filter(FIELD<float>* f,int i,int j,float& gx,float& gy)	//compute gradient of f[i][j] in gx,gy
{									//by using smoothing on a N-pixel neighborhood
  gx = gy = 0;  float ci = 0, cj = 0; float wsi = 0, wsj = 0;
  
  for(int ii=-N;ii<=N;ii++)
      for(int jj=-N;jj<=N;jj++)
      {
        ci += w[N+ii][N+jj]*ii*f->value(i+ii,j+jj);
        cj += w[N+ii][N+jj]*jj*f->value(i+ii,j+jj);
        wsi += w[N+ii][N+jj]*ii*ii;
        wsj += w[N+ii][N+jj]*jj*jj;
      }
  
  gx = ci/wsi; gy = cj/wsj;                             //normalize gradient
  float r = sqrt(gx*gx+gy*gy);
  gx /= r; gy /= r;
}
       


void gradient(FIELD<float>* f,int i,int j,float& gx,float& gy)	//compute gradient of f[i][j] in gx,gy
{
  gx = gy = 0;  float ci = 0, cj = 0; float wsi = 0, wsj = 0;
  
  const int N=0;
  for(int ii=0;ii<=N;ii++)
      for(int jj=-N;jj<=N;jj++)
      {
        ci += f->value(i+ii+1,j+jj)-f->value(i+ii,j+jj);
        cj += f->value(i+ii,j+jj+1)-f->value(i+ii,j+jj);
      }
  const float SZ = 2*N+1;

  gx = ci/SZ; gy = cj/SZ;                             //normalize gradient
  float r = sqrt(gx*gx+gy*gy);
  if (r>0.00001) { gx /= r; gy /= r; }
}
    


void compute_gradient(FIELD<float>* f,FIELD<float>*& gx,FIELD<float>*& gy)
{								//compute gradient of 'f' in 'gx','gy'
  gx = new FIELD<float>(f->dimX(),f->dimY()); *gx = 0;			
  gy = new FIELD<float>(f->dimX(),f->dimY()); *gy = 0;

  int i,j;
  if (N)							//N>0? Use gradient-computation by smoothing
    for(i=N+1;i<f->dimX()-N-1;i++)				//with a filter-size of N pixels
       for(j=N+1;j<f->dimY()-N-1;j++)
          gradient_filter(f,i,j,gx->value(i,j),gy->value(i,j));
  else								//N=0? Use no smoothing, compute gradient directly
     for(i=N+1;i<f->dimX()-N-1;i++)				//by central differences.
       for(j=N+1;j<f->dimY()-N-1;j++)
          gradient(f,i,j,gx->value(i,j),gy->value(i,j));
}



void draw(FIELD<float>& f)
{
   static unsigned char buf[1000*1000*3];
   float m,M,avg; f.minmax(m,M,avg);
   const float* d = f.data(); int j=0;
   for(int i=0;i<f.dimY();i++)
     for(const float *s=d+f.dimX()*i, *e=s+f.dimX();s<e;s++)
     {
       float r,g,b,v = ((*s)-m)/(M-m); 
       v = MAX(v,0); 
       if (v>M) { r=g=b=1; } else v = MIN(v,1);
       float2rgb(v,r,g,b);
       buf[j++] = (unsigned char)(int)(255*r);
       buf[j++] = (unsigned char)(int)(255*g);
       buf[j++] = (unsigned char)(int)(255*b);
     }
   glPixelStorei(GL_UNPACK_ALIGNMENT,1);	
   glDrawPixels(f.dimX(),f.dimY(),GL_RGB,GL_UNSIGNED_BYTE,buf);
}


void draw(IMAGE<float>& f)
{
   static unsigned char buf[1000*1000*3];
   float mr,Mr,avg,mg,Mg,mb,Mb; 
   f.r.minmax(mr,Mr,avg); f.g.minmax(mg,Mg,avg); f.b.minmax(mb,Mb,avg);
   const float *dr = f.r.data(), *dg = f.g.data(), *db = f.b.data(); 
   int j=0;
   for(int i=0;i<f.dimY();i++)
     for(const float *sr=dr+f.dimX()*i, 
                     *sg=dg+f.dimX()*i,
                     *sb=db+f.dimX()*i,*e=sr+f.dimX();sr<e;sr++,sg++,sb++)
     {
       float r = ((*sr)-mr)/(Mr-mr),
             g = ((*sg)-mg)/(Mg-mg),
             b = ((*sb)-mb)/(Mb-mb);
       r = MAX(r,0); g = MAX(g,0); b = MAX(b,0);         
       r = MIN(r,1); g = MIN(g,1); b = MIN(b,1);
       buf[j++] = (unsigned char)(int)(255*r);
       buf[j++] = (unsigned char)(int)(255*g);
       buf[j++] = (unsigned char)(int)(255*b);
     }
   glPixelStorei(GL_UNPACK_ALIGNMENT,1);	
   glDrawPixels(f.dimX(),f.dimY(),GL_RGB,GL_UNSIGNED_BYTE,buf);
}

void draw(FIELD<int>& f)
{
   static unsigned char buf[1000*1000*3];
 
   const int* d = f.data(); int j=0;
   for(int i=0;i<f.dimY();i++)
     for(const int *s=d+f.dimX()*i, *e=s+f.dimX();s<e;s++)
     {
       switch(*s)
       {
         case FLAGS::ALIVE:	    buf[j++] = 255; buf[j++] = 0;   buf[j++] = 0;   break;
	 case FLAGS::NARROW_BAND:   buf[j++] = 0;   buf[j++] = 255; buf[j++] = 0;   break;
	 case FLAGS::FAR_AWAY:      buf[j++] = 255; buf[j++] = 255; buf[j++] = 255; break;
	 case FLAGS::EXTREMUM:	    buf[j++] = 0;   buf[j++] = 0;   buf[j++] = 255; break;
         default:                   buf[j++] = 0;   buf[j++] = 0;   buf[j++] = 0;
       }
     }
   glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   glDrawPixels(f.dimX(),f.dimY(),GL_RGB,GL_UNSIGNED_BYTE,buf);
}




void display(void) 
{
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW); glLoadIdentity();
	draw(*rgb_image);
	glFlush(); glutSwapBuffers();
}

void reshape(int w, int h) 
{
 	glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
	glMatrixMode(GL_PROJECTION);  
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
}
