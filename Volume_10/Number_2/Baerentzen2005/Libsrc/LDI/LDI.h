#ifndef __LDI_H
#define __LDI_H

#include <vector>
#include <list>
#include "CGLA/Vec3f.h"
#include "CGLA/Mat4x4f.h"
#include "CGLA/Vec4f.h"
#include "Components/Grid2D.h"
#include "PointRecord.h"


namespace LDI
{

	class LDILayer
	{
		CMP::Grid2D<CGLA::Vec3f> normal_buffer;
		CMP::Grid2D<CGLA::Vec4f> colour_buffer;
		CMP::Grid2D<float> depth_buffer;

	public:

		LDILayer(int xdim, int ydim):
			normal_buffer(xdim, ydim),
			colour_buffer(xdim, ydim),
			depth_buffer(xdim, ydim) {}

		float* get_normal_buffer()
		{
			return reinterpret_cast<float*>(&normal_buffer(0,0));
		}
		float* get_colour_buffer()
		{
			return reinterpret_cast<float*>(&colour_buffer(0,0));
		}
		float* get_depth_buffer()
		{
			return reinterpret_cast<float*>(&depth_buffer(0,0));
		}

		bool convert_to_points(const CGLA::Mat4x4f& itransf, 
													 std::vector<PointRecord>& points);
	};

	struct Run
	{
		CGLA::Vec3f pos;
		CGLA::Vec3f dir;
		std::vector<PointRecord> pts;
	};
	
	class LDImage
	{
		int xdim, ydim;
		std::list<LDILayer> layers;
		CGLA::Mat4x4f itransf;
		CGLA::Mat4x4f itransf_noscale;

	public:

		int no_layers() const {return layers.size();}

		void convert_to_points(std::vector<PointRecord>& points)
		{
			std::list<LDILayer>::iterator i;
			for(i=layers.begin();i != layers.end();++i)
				(*i).convert_to_points(itransf, points);
		}
		
		LDILayer* add_layer()
		{
			LDILayer l(xdim, ydim);
			layers.push_back(l);
			return &layers.back();
		}

		LDImage(int _xdim, int _ydim, 
						const CGLA::Mat4x4f& _itransf,
						const CGLA::Mat4x4f& _itransf_noscale): 
			xdim(_xdim), ydim(_ydim), 
			itransf(_itransf), 
			itransf_noscale(_itransf_noscale)
			{}
	};

	class LDISet
	{
		std::list<LDImage> ldis;
	public:
		int ldi_size;	
		float unit_scale;
		CGLA::Vec3f dims;
		CGLA::Vec3f orig_dims;
		CGLA::Vec3f orig_offs;

		LDISet(int _ldi_size): ldi_size(_ldi_size) {}

		int no_ldis() const {return ldis.size();}

		void convert_to_points(std::vector<PointRecord>& points)
		{
			std::list<LDImage>::iterator i;
			for(i=ldis.begin();i != ldis.end();++i)
				(*i).convert_to_points(points);
		}
		LDImage* add_ldi(int _xdim, int _ydim, 
										 const CGLA::Mat4x4f& _itransf,
										 const CGLA::Mat4x4f& _itransf_noscale)
		{
			LDImage l(_xdim, _ydim, _itransf, _itransf_noscale);
			ldis.push_back(l);
			return &ldis.back();
		}

	};
}

#endif
