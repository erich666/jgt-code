#ifndef IMAGE_H
#define IMAGE_H


#include "field.h"
#include <fstream.h>

template <class T> class IMAGE
	{
	public:
			IMAGE(int x=0,int y=0):r(x,y),g(x,y),b(x,y) {}
                        IMAGE(const IMAGE& i):r(i.r),g(i.g),b(i.b)  {}
        void            setValue(int,int,T);
	void		size(int i,int j)		{ r.size(i,j); g.size(i,j); b.size(i,j); } 
	void		size(const FIELD<T>& f)         { r.size(f); g.size(f); b.size(f); }	
	
	FIELD<T>	r,g,b;
	int		dimX() const			{ return r.dimX(); }
	int		dimY() const			{ return r.dimY(); }
        void            normalize()                    { r.normalize(); g.normalize(); b.normalize(); }
	
	static IMAGE*	read(char*);

        private:

        static IMAGE*   readBMP(char*);
        static unsigned long
                        getLong(fstream&);
        static unsigned short
                        getShort(fstream&);
	};
		
	
template <class T> IMAGE<T>* IMAGE<T>::read(char* fname)
{
   switch(FIELD<T>::fileType(fname))
   {
   case FIELD<T>::BMP:   return readBMP(fname);
   default:    return 0;
   }
}

template <class T> inline void IMAGE<T>::setValue(int i,int j,T v)
{ r.value(i,j) = g.value(i,j) = b.value(i,j) = v; }

template <class T> IMAGE<T>* IMAGE<T>::readBMP(char* fname)
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
   IMAGE<float>* f = new IMAGE<float>(numCols,numRows);
   float *rd = f->r.data(), *gd = f->g.data(), *bd = f->b.data();
   unsigned char ch;
   for(unsigned int row=0;row<numRows;row++)               //for every row     
   {
      for(unsigned int col=0;col<numCols;col++)
      {
         if (bpp==3)                                       //read data as RGB colors
         {  
            unsigned char R,G,B; inf.get(B); inf.get(G); inf.get(R);
            *rd++ = R; *gd++ = G; *bd++ = B;
         }
         else                                              //read data as 8-bit luminance
         {  inf.get(ch); *rd++ = ch; *gd++ = ch; *bd++ = ch; }
      }
      for(unsigned int k=0;k<numPadBytes;k++) inf>>ch;     //skip pad bytes at end of row
   }
   inf.close();
   return f;
}


     

template <class T> unsigned long IMAGE<T>::getLong(fstream& inf)
{
   unsigned long ip; char ic;
   unsigned char uc;
   inf.get(ic); uc = ic; ip = uc;
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<8);
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<16);
   inf.get(ic); uc = ic; ip |= ((unsigned long)uc <<24);
   return ip;
}

template <class T> unsigned short IMAGE<T>::getShort(fstream& inf)
{
   char ic; unsigned short ip;
   inf.get(ic); ip = ic;
   inf.get(ic); ip |= ((unsigned short)ic << 8);
   return ip;
}



#endif

