#ifndef __GRID2D_H
#define __GRID2D_H

#include <vector>
#include "CGLA/Vec2i.h"

namespace Components
{

	template<class T>
	class Grid2D
	{
		int XDIM, YDIM;
		std::vector<T> pixels;

	public:
		
		Grid2D(int i, int j, const T& val): XDIM(i), YDIM(j), pixels(i*j, val) {}
		Grid2D(int i, int j): XDIM(i), YDIM(j), pixels(i*j) {}
		
		const T& operator()(int i, int j) const 
		{
		  assert(i>=0 && i< XDIM);
		  assert(j>=0 && j< YDIM);
		  return pixels[j*XDIM+i];
		}

		T& operator()(int i, int j) 
		{
		  assert(i>=0 && i< XDIM);
		  assert(j>=0 && j< YDIM);
		  return pixels[j*XDIM+i];
		}

		const T& operator()(const CGLA::Vec2i& p) const 
		{
		  assert(p[0]>=0 && p[0]< XDIM);
		  assert(p[1]>=0 && p[1]< YDIM);
		  return pixels[p[1]*XDIM+p[0]];
		}

		T& operator()(const CGLA::Vec2i& p) 
		{
		  assert(p[0]>=0 && p[0]< XDIM);
		  assert(p[1]>=0 && p[1]< YDIM);
		  return pixels[p[1]*XDIM+p[0]];
		}

		int get_xdim() const {return XDIM;}
		int get_ydim() const {return YDIM;}
		
	};
}
namespace CMP = Components;
#endif
