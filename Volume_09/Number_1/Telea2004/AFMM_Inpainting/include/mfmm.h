#ifndef MFMM_H
#define MFMM_H

#include "fmm.h"
#include "image.h"

//ModifiedFastMarchingMethod: Adds inpainting capabilities to the FMM.
//
//
//


class 	ModifiedFastMarchingMethod : public FastMarchingMethod
	{
	public:
			ModifiedFastMarchingMethod(FIELD<float>* f,
						   FLAGS*,
						   IMAGE<float>* img,
						   FIELD<float>* gx,
						   FIELD<float>* gy,
						   FIELD<float>* dst,
						   int B_radius,
						   int dst_weighting,
						   int lev_weighting,
						   int);
						   			//Ctor
		int     execute(int&,int&,float=INFINITY);		//Enh inherited to compute the image field
									
	protected:
	
		void	add_to_narrowband(int,int,int,int);		//Enh inherited to update 'count'

	private:

		IMAGE<float>*	image;					//Image to inpaint
									//
		FIELD<float>   *dist;					//Distance field, needed for inpainting
		FIELD<float>   *grad_x,*grad_y;				//Gradient of 'dist' field, needed for inpainting
		int		B_radius;				//Radius of inpainting-neighborhood, in pixels
		int		dst_weighting;				//Flag telling if we use dist-based weighting (def: 1)
		int		lev_weighting;				//Flag telling if we use level-based weighting (def: 1)
			
		int diffuse();
	};	



#endif				



