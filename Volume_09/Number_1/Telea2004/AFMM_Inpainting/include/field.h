#ifndef FIELD_H
#define FIELD_H


#include <stdio.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <io.h>



class FLAGS;

template <class T> class FIELD
	{
	public:

                enum FILE_TYPE {
                        BMP, PGM, VTK, ASCII, UNKNOWN 
                        };
                        

			FIELD(int=0,int=0);
			FIELD(const FIELD&);
	       	       ~FIELD();
	  T&            value(int,int);				//value at (i,j)
          const T&      value(int,int) const;			//const version of above
	  const T	value(float,float) const;		//bilinearly interpolated value anywhere inside field 
	  T		gradnorm(int,int) const;		//norm of grad at (i,j)
	  T*	        data()					{ return v;  }
          const T*      data() const            		{ return v;  }
	  int		dimX()			{ return nx; }	//number of columns
	  int		dimY()			{ return ny; }	//number of rows
	  int		dimX() const		{ return nx; }	//number of columns
	  int		dimY() const		{ return ny; }	//number of rows
	  FIELD&	operator=(FIELD&);			//assignment op
	  FIELD&	operator=(T);				//assignment op
          FIELD&        operator+=(const FIELD&);               //addition op
	  FIELD&        operator-=(const FIELD&);		//subtraction op
          FIELD&        operator/=(const FIELD&);		//division op
	  void		gradnorm(FIELD&) const;			//norm of grad as field
	  void		minmax(T&,T&,T&) const;			//min, max, avg for field
	  void		normalize();				//normalize this between 0 and 1
	  FIELD&	operator*=(T);				//multiply field by scalar
	  void		write(char*);				//write field to VTK struct points data file
	  void		writeGrid(char*);			//write field to VTK struct grid data file
	  void		writePPM(char*) const;			//write field to PPM RGB file
	  void		writePGM(char*) const;			//write field to PGM grayscale file
	  
	  static FIELD* read(char*);				//read field from data file in various formats
          static FILE_TYPE                                      //get file type we could read from
                        fileType(char*);        
	  void		size(const FIELD&);
	  
	private:

	  static FIELD*	readVTK(char*);				//read field from VTK scalar data file	
	  static FIELD* readPGM(char*);				//read field from PGM binary file
	  static FIELD* readASCII(char*);			//read field from plain ASCII data file
          static FIELD* readBMP(char*);                         //read field from BMP image file
          static unsigned long
                        getLong(fstream&);
          static unsigned short
                        getShort(fstream&);
	  int		nx,ny;					//nx = ncols, ny = nrows
	  T*		v;
	};



template <class T> class VFIELD
	{
	public:
			VFIELD(int x=0,int y=0):v0(x,y),v1(x,y) {}
	void		setValue(int,int,T*);
        void            write(const char*) const;
	void		write(const char*,FLAGS&) const;
	void		size(int i,int j)		{ v0.size(i,j); v1.size(i,j); } 
	void		size(const FIELD<T>& f)         { v0.size(f); v1.size(f); }	
	
	FIELD<T>	v0;
	FIELD<T>	v1;
	int		dimX() const			{ return v0.dimX(); }
	int		dimY() const			{ return v0.dimY(); }
	
	static VFIELD*	read(char*);
	};
		
		


//---------------------------------------------------------------










template <class T> inline FIELD<T>::FIELD(int nx_,int ny_): nx(nx_),ny(ny_),v((nx_*ny_)? new T[nx_*ny_] : 0)
{  }

template <class T> inline FIELD<T>::FIELD(const FIELD<T>& f): nx(f.nx),ny(f.ny),v((f.nx*f.ny)? new T[f.nx*f.ny] : 0)
{  if (nx*ny) memcpy(v,f.v,nx*ny*sizeof(T));  }

template <class T> inline T& FIELD<T>::value(int i,int j)
{
  i = (i<0) ? -i : (i>=nx) ? 2*nx-i-1 : i;
  j = (j<0) ? -j : (j>=ny) ? 2*ny-j-1 : j; 
  return *(v+j*nx+i);
}

template <class T> inline const T FIELD<T>::value(float i,float j) const
{
  int       ii = int(i), jj      = int(j);				//get cell in which the floating-point

  T f1 = value(ii,jj)*(1+jj-j)   + value(ii,jj+1)*(j-jj);
  T f2 = value(ii+1,jj)*(1+jj-j) + value(ii+1,jj+1)*(j-jj);
  
  return (f2-f1)*(i-ii)+f1; 


  //const T* pij = v+jj*nx+ii;						//falls. We shall then bilinearly interpolate
  //T f1 = (*pij)*(1+jj-j)     + (*(pij+nx))*(j-jj);			//between the 4 cell's vertex values. 
  //T f2 = (*(pij+1))*(1+jj-j) + (*(pij+nx+1))*(j-jj);			//We write the expressions directly, and not in 
									//terms of FIELD::value(), since it is faster
  //return (f2-f1)*(i-ii)+f1;						
}   



template <class T> inline const T& FIELD<T>::value(int i,int j) const
{
  i = (i<0) ? i : (i>=nx) ? 2*nx-i-1 : i;
  j = (j<0) ? j : (j>=ny) ? 2*ny-j-1 : j; 
  return *(v+j*nx+i);
}


template <class T> inline T FIELD<T>::gradnorm(int i,int j) const
{
   T ux = value(i+1,j)-value(i-1,j);
   T uy = value(i,j+1)-value(i,j-1);
   return (ux*ux+uy*uy)/4;
}

template <class T> inline FIELD<T>::~FIELD()
{  delete[] v;  }

template <class T> inline void FIELD<T>::size(const FIELD& f)
{
   delete[] v;
   nx = f.nx; ny = f.ny;
   v = (nx*ny) ? new T[nx*ny] : 0;
}

template <class T> FIELD<T>& FIELD<T>::operator=(FIELD& f)
{
   if (nx!=f.nx || ny!=f.ny)
	size(f);
   if (nx*ny) memcpy(v,f.v,nx*ny*sizeof(T));
   return *this;
}   

template <class T> FIELD<T>& FIELD<T>::operator+=(const FIELD& f)
{
   if (f.dimX()==dimX() && f.dimY()==dimY())
   { 
      const T* fptr = f.data();
      for(T *vptr=v,*vend=v+nx*ny;vptr<vend;vptr++,fptr++)
         (*vptr) += (*fptr);
   }
   return *this;
}

template <class T> FIELD<T>& FIELD<T>::operator-=(const FIELD& f)
{
   if (f.dimX()==dimX() && f.dimY()==dimY())
   { 
      const T* fptr = f.data();
      for(T *vptr=v,*vend=v+nx*ny;vptr<vend;vptr++,fptr++)
         (*vptr) -= (*fptr);
   }
   return *this;
}

template <class T> FIELD<T>& FIELD<T>::operator/=(const FIELD& f)
{
   if (f.dimX()==dimX() && f.dimY()==dimY())
   { 
      const T* fptr = f.data();
      for(T *vptr=v,*vend=v+nx*ny;vptr<vend;vptr++,fptr++)
	if (fabs(*fptr)>1.0e-7)
           (*vptr) /= (*fptr);
   }
   return *this;
}


template <class T> FIELD<T>& FIELD<T>::operator=(T val)
{
   for(T* vptr=v,*vend=v+nx*ny;vptr<vend;vptr++)
      (*vptr) = val;
   return *this;
}   

template <class T> void FIELD<T>::gradnorm(FIELD& f) const
{
   f.size(*this);

   for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++)
         f.value(i,j) = gradnorm(i,j);
}

template <class T> void FIELD<T>::minmax(T& m,T& M,T& a) const
{
   const float INFINITY_2 = INFINITY/2;

   if (nx*ny<2) { m = M = a = 0; return; }
   m = v[0];
   M = -T(INFINITY);
   a = 0;

   for(T* vptr = v,*vend = v+nx*ny;vptr<vend;vptr++)
   {
	if (m > *vptr) m = *vptr;
    	if (M < *vptr && *vptr < INFINITY_2) M = *vptr;
	a += *vptr;
   }
   a /= nx*ny;
}      

template <class T> void FIELD<T>::normalize()
{
   const float INFINITY_2 = INFINITY/2;

   float m,M,a,d; minmax(m,M,a); 
   d = (M-m>1.0e-5)? M-m : 1;

   for(T* vptr = v,*vend = v+nx*ny;vptr<vend;vptr++)
   {
	float v = *vptr;
	*vptr = (v>M)? 1 : (v-m)/d;
   }
}      



template <class T> FIELD<T>& FIELD<T>::operator*=(T f)
{
   for(T* vptr = v,*vend = v+nx*ny;vptr<vend;vptr++)
      *vptr *= f;
   return *this;
}


template <class T> FIELD<T>* FIELD<T>::read(char* fname)
{
   switch(fileType(fname))
   {
      case VTK:   return readVTK(fname);
      case BMP:   return readBMP(fname);
      case ASCII: return readASCII(fname);
      case PGM:   return readPGM(fname);
      default:    return 0;
   }
}   
   
   
template <class T> FIELD<T>::FILE_TYPE FIELD<T>::fileType(char* fname)
{
   FILE* fp = fopen(fname,"r");
   if (!fp) return UNKNOWN;
   
   char c1,c2;
   if (fscanf(fp,"%c%c",&c1,&c2)!=2) return UNKNOWN;
   fclose(fp);
   
   if (c1=='#') return VTK;
   if (c1=='P' && c2=='5') return PGM;
   if (c1=='B' && c2=='M') return BMP;
   return ASCII;
}   
   
   

     

template <class T> unsigned long FIELD<T>::getLong(fstream& inf)
{
   unsigned long ip; char ic;
   unsigned char uc;
   inf.get(ic); uc = ic; ip = uc;
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<8);
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<16);
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<24);
   return ip;
}

template <class T> unsigned short FIELD<T>::getShort(fstream& inf)
{
   char ic; unsigned short ip;
   inf.get(ic); ip = ic;
   inf.get(ic); ip |= ((unsigned short)ic << 8);
   return ip;
}


template <class T> FIELD<T>* FIELD<T>::readBMP(char* fname)
{
   fstream inf;
   inf.open(fname, ios::in|ios::binary);
   if (!inf) return 0;
   char ch1,ch2;
   inf.get(ch1); inf.get(ch2);                             //read BMP header
   unsigned long  fileSize = getLong(inf);      
   unsigned short res1     = getShort(inf);
   unsigned short res2     = getShort(inf);
   unsigned long  offBits  = getLong(inf);
   unsigned long  hdrSize  = getLong(inf);
   unsigned long  numCols  = getLong(inf);     
   unsigned long  numRows  = getLong(inf);     
   unsigned short planes   = getShort(inf);
   unsigned short bytesPix = getShort(inf);                //8 or 24
   unsigned long  compr    = getLong(inf); 
   unsigned long  imgSize  = getLong(inf);
   unsigned long  xPels    = getLong(inf);
   unsigned long  yPels    = getLong(inf);
   unsigned long  lut      = getLong(inf);
   unsigned long  impCols  = getLong(inf);
   int bpp = bytesPix/8;                                   //1 or 3                        
   unsigned int nBytesInRow  = ((bpp*numCols+3)/4)*4;
   unsigned int numPadBytes  = nBytesInRow - bpp*numCols; 
   FIELD<float>* f = new FIELD<float>(numCols,numRows);
   float* data = f->data();
   unsigned char ch;
   for(unsigned int row=0;row<numRows;row++)               //for every row     
   {
      for(unsigned int col=0;col<numCols;col++)
      {
         if (bpp==3)                                       //read data as RGB 'luminance'
         {  
            unsigned char r,g,b; inf.get(b); inf.get(g); inf.get(r);
            *data++ = (float(r)+float(g)+float(b))/3;
         }
         else                                              //read data as 8-bit luminance
         {  inf.get(ch); *data++ = ch; }
      }
      for(unsigned int k=0;k<numPadBytes;k++) inf>>ch;     //skip pad bytes at end of row
   }
   inf.close();
   return f;
}



template <class T> FIELD<T>* FIELD<T>::readVTK(char* fname)	//read VTK scalar file into this
{
   FILE* fp = fopen(fname,"r");
   if (!fp) return 0;

   char buf[100]; 

   FIELD<T>* f = 0;

   for(;fscanf(fp,"%s",buf)==1;)
   {
      if (!strcmp(buf,"DIMENSIONS"))
      {
	int dimX,dimY;
	fscanf(fp,"%d%d",&dimX,&dimY);
	f = new FIELD<T>(dimX,dimY);
      }	
	
      if (!strcmp(buf,"LOOKUP_TABLE"))
      {
	 fscanf(fp,"%*s");
	 break;
      }
   }

   for(T* d = f->data();fscanf(fp,"%f",d)==1;d++);

   fclose(fp);  				
   return f;
}



template <class T> FIELD<T>* FIELD<T>::readPGM(char* fname)	//read VTK scalar file into this
{
   FILE* fp = fopen(fname,"r"); if (!fp) return 0;

   const int SIZE = 1024;
   char buf[SIZE]; int dimX,dimY,range;
   fscanf(fp,"%*s");				//skip "P5" header

   for(;;)
   {
     fscanf(fp,"%s",buf);			//get dimX or #comment
     if (buf[0]=='#') fgets(buf,SIZE,fp); 
        else { dimX = atoi(buf); break; }
   }
   for(;;)
   {
     fscanf(fp,"%s",buf);			//get dimY or #comment
     if (buf[0]=='#') fgets(buf,SIZE,fp); 
        else { dimY = atoi(buf); break; }
   }
   for(;;)
   {
     fscanf(fp,"%s",buf);			//get range or #comment
     if (buf[0]=='#') fgets(buf,SIZE,fp); 
        else { range = atoi(buf); break; }
   }
   

   FIELD<T>* f = new FIELD<T>(dimX,dimY);
   int bb = SIZE; fgets(buf,SIZE,fp);
  
   for(T *d = f->data(),*end=d+dimX*dimY;d<end;d++)		//read the binary data into the field
   {								//be careful: buf is a char, we first need
	if (bb==SIZE) { fread(buf,SIZE,1,fp); bb=0; }		//to convert the read bytes to unsigned char and then assign
	*d = (unsigned char)buf[bb++];				//to the field!
   }

   fclose(fp);  				
   return f;
}



template <class T> FIELD<T>* FIELD<T>::readASCII(char* fname)	//read plain ASCII file into this
{
   FILE* fp = fopen(fname,"r");
   if (!fp) return 0;

   FIELD<T>* f = 0;

   int dimX,dimY,dimZ;
   fscanf(fp,"%d%d%d",&dimX,&dimY,&dimZ);
   f = new FIELD<T>(dimX,dimY);

   for(T* d = f->data();fscanf(fp,"%f",d)==1;d++);

   fclose(fp);  				
   return f;
}




template <class T> void FIELD<T>::write(char* fname)
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   fprintf(fp,"# vtk DataFile Version 2.0\n"
	      "vtk output\n"
	      "ASCII\n"
	      "DATASET STRUCTURED_POINTS\n"
	      "DIMENSIONS %d %d 1\n"
	      "SPACING 1 1 1\n"
	      "ORIGIN 0 0 0\n"
	      "POINT_DATA %d\n"
	      "SCALARS scalars float\n"
	      "LOOKUP_TABLE default\n",
	      nx,ny,nx*ny);

   for(T* vend=v+nx*ny,*vptr=v;vptr<vend;vptr++)
      fprintf(fp,"%f\n",float(*vptr));

   fclose(fp);
}	


template <class T> void FIELD<T>::writeGrid(char* fname)
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   fprintf(fp,"# vtk DataFile Version 2.0\n"
	      "vtk output\n"
	      "ASCII\n"
	      "DATASET STRUCTURED_GRID\n"
	      "DIMENSIONS %d %d 1\n"
	      "POINTS %d int\n",
	      nx,ny,nx*ny);

   for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++)
         fprintf(fp,"%d %d 0\n",i,j);

   fprintf(fp,"POINT_DATA %d\n"
	      "SCALARS scalars float\n"
	      "LOOKUP_TABLE default\n",
	      nx*ny);

   for(T* vend=v+nx*ny,*vptr=v;vptr<vend;vptr++)
      fprintf(fp,"%f\n",float(*vptr));

   fclose(fp);
}	


template <class T> void FIELD<T>::writePGM(char* fname) const
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   float m,M; T m_,M_,avg_; minmax(m_,M_,avg_);
   m = m_; M = M_; 

   const int SIZE = 3000;
   unsigned char buf[SIZE];
   int bb=0;

   fprintf(fp,"P5 %d %d 255\n",dimX(),dimY());
   for(const T* vend=data()+dimX()*dimY(),*vptr=data();vptr<vend;vptr++)
   {
      float v = ((*vptr)-m)/(M-m); v = MAX(v,0); 
      if (v>M) v=1; else v = MIN(v,1);
      buf[bb++] = (unsigned char)(int)(v*255);
      if (bb==SIZE)
      {  fwrite(buf,1,SIZE,fp); bb = 0; }
   }
   if (bb) fwrite(buf,1,bb,fp);

   fclose(fp);
}	


template <class T> void FIELD<T>::writePPM(char* fname) const
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   float m,M; T m_,M_,avg_; minmax(m_,M_,avg_);
   m = m_; M = M_; 

   const int SIZE = 3000;
   unsigned char buf[SIZE];
   int bb=0;

   fprintf(fp,"P6 %d %d 255\n",dimX(),dimY());
   for(const T* vend=data()+dimX()*dimY(),*vptr=data();vptr<vend;vptr++)
   {
      float r,g,b,v = ((*vptr)-m)/(M-m); 
      v = max(v,0); 
      if (v>M) { r=g=b=1; } else v = min(v,1);
      float2rgb(v,r,g,b);
      
      buf[bb++] = (unsigned char)(int)(r*255);
      buf[bb++] = (unsigned char)(int)(g*255);
      buf[bb++] = (unsigned char)(int)(b*255);
      if (bb==SIZE)
      {  fwrite(buf,1,SIZE,fp); bb = 0; }
   }
   if (bb) fwrite(buf,1,bb,fp);

   fclose(fp);
}	

//------------------  VFIELD  -----------------------------------


template <class T> inline void VFIELD<T>::setValue(int i,int j,T* val)
{
   v0.value(i,j) = val[0];
   v1.value(i,j) = val[1];
}

template <class T> VFIELD<T>* VFIELD<T>::read(char* fname)
{
   FILE* fp = fopen(fname,"r");
   if (!fp) return 0;

   VFIELD<T>* f = 0;

   char buf[100]; 
   for(;fscanf(fp,"%s",buf)==1;)
   {
      if (!strcmp(buf,"DIMENSIONS"))
      {
	int dimX,dimY;
	fscanf(fp,"%d%d",&dimX,&dimY);
	f = new VFIELD<T>(dimX,dimY);
      }	
	
      if (!strcmp(buf,"VECTORS"))
      {
	 fscanf(fp,"%*s%*s");
	 break;
      }
   }

   for(T* d0=f->v0.data(),*d1=f->v1.data();fscanf(fp,"%f%f%*f",d0,d1)>1;d0++,d1++);

   fclose(fp);  				
   return f;
}

template <class T> void VFIELD<T>::write(const char* fname,FLAGS& f) const
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   fprintf(fp,"# vtk DataFile Version 2.0\n"
	      "vtk output\n"
	      "ASCII\n"
	      "DATASET STRUCTURED_POINTS\n"
	      "DIMENSIONS %d %d 1\n"
	      "SPACING 1 1 1\n"
	      "ORIGIN 0 0 0\n"
	      "POINT_DATA %d\n"
	      "VECTORS vectors float\n",
	      dimX(),dimY(),dimX()*dimY());

   for(int j=0;j<dimY();j++)
      for(int i=0;i<dimX();i++)
         if (f.alive(i,j))
            fprintf(fp,"0 0 0\n");
         else
            fprintf(fp,"%f %f 0\n",v0.value(i,j),v1.value(i,j));  

   fclose(fp);
}	


template <class T> void VFIELD<T>::write(const char* fname) const
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   fprintf(fp,"# vtk DataFile Version 2.0\n"
	      "vtk output\n"
	      "ASCII\n"
	      "DATASET STRUCTURED_POINTS\n"
	      "DIMENSIONS %d %d 1\n"
	      "SPACING 1 1 1\n"
	      "ORIGIN 0 0 0\n"
	      "POINT_DATA %d\n"
	      "VECTORS vectors float\n",
	      dimX(),dimY(),dimX()*dimY());

   for(int j=0;j<dimY();j++)
      for(int i=0;i<dimX();i++)
         fprintf(fp,"%f %f 0\n",v0.value(i,j),v1.value(i,j));  

   fclose(fp);
}	


#endif
