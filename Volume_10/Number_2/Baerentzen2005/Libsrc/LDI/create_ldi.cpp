#include <cfloat>
#include <GLEW/glew.h>
#include "Graphics/DepthPeeler.h"
#include "LDI.h"
#include "create_ldi.h"

namespace LDI
{
	using namespace CGLA;
	using namespace CMP;
	using namespace CMN;
	using namespace GFX;
	using namespace std;

	const Vec3f directions[3] = 
		{
			Vec3f(1,0,0),
			Vec3f(0,1,0),
			Vec3f(0,0,1)
		};
	
	inline const Mat4x4f create_transf(const Vec3f& d0)
	{
		Vec3f d1,d2;
		int D = -1;
		if(fabs(d0[0])>0.99) D=0;
		if(fabs(d0[1])>0.99f) D=1;
		if(fabs(d0[2])>0.99f) D=2;
		orthogonal(d0, d1, d2);
		return Mat4x4f(Vec4f(d1,0),Vec4f(d2,0),Vec4f(d0,0),Vec4f(0,0,0,1));
	}

	void draw_with_normals_as_color()
	{
		Vec3f v0(0,0,0), v1(1,0,0), v2(1,1,1);
		Vec3f v3(0,0,1), v4(1,0,0.6), v5(1,1,0);

		Vec3f n1 = Vec3f(0.5) + 0.5 * normalize(cross(v1-v0, v2-v0));
		Vec3f n2 = Vec3f(0.5) + 0.5 * normalize(cross(v4-v3, v5-v3));	

		glColor3fv(n1.get());
		glBegin(GL_TRIANGLES);
		glVertex3fv(v0.get());
		glVertex3fv(v1.get());
		glVertex3fv(v2.get());
		glEnd();

		glColor3fv(n2.get());
		glBegin(GL_TRIANGLES);
		glVertex3fv(v3.get());
		glVertex3fv(v4.get());
		glVertex3fv(v5.get());
		glEnd();
	}

	void draw()
	{
		Vec3f v0(0,0,0), v1(1,0,0), v2(1,1,1);
		Vec3f v3(0,0,1), v4(1,0,0.6), v5(1,1,0);

		glColor3f(1,0,0);
		glBegin(GL_TRIANGLES);
		glVertex3fv(v0.get());
		glVertex3fv(v1.get());
		glVertex3fv(v2.get());
		glEnd();

		glColor3f(0,0,1);
		glBegin(GL_TRIANGLES);
		glVertex3fv(v3.get());
		glVertex3fv(v4.get());
		glVertex3fv(v5.get());
		glEnd();
	}
	
	void create_ldi(LDISet& ldi_set)
	{
		int vp[4];
		glGetIntegerv(GL_VIEWPORT, vp);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		
		const int no_dir = 3;
		Mat4x4f transf[no_dir];
		for(int D=0;D<no_dir;++D)
			transf[D] = create_transf(directions[D]);
		
		Vec3f p0_orig(0,0,0);
		Vec3f p7_orig(1,1,1);

		Vec3f dim_orig = (p7_orig-p0_orig);
		ldi_set.unit_scale = dim_orig.length()/ldi_set.ldi_size;
		
		Vec3f tmp_dims = dim_orig/ldi_set.unit_scale;
		ldi_set.dims = Vec3f(ceil(tmp_dims[0]),
												 ceil(tmp_dims[1]),
												 ceil(tmp_dims[2]));
		ldi_set.orig_dims = dim_orig;
		ldi_set.orig_offs = p0_orig;

		for(int D=0;D<no_dir;++D)
			{
				Vec3f p0(transf[D].mul_3D_point(p0_orig));
				Vec3f p7(p0);
				for(int i=1;i<8;++i)
					{
 						const Vec3f p_orig(i&1?p7_orig[0]:p0_orig[0],
															 i&2?p7_orig[1]:p0_orig[1],
															 i&4?p7_orig[2]:p0_orig[2]);
						const Vec3f p = transf[D].mul_3D_point(p_orig);
						p0 = v_min(p0, p);
						p7 = v_max(p7, p);
					}
				const Vec3f dim = (p7-p0)/ldi_set.unit_scale;
				const int width  = static_cast<int>(ceil(fabs(dim[0])));
				const int height = static_cast<int>(ceil(fabs(dim[1])));

				Mat4x4f imat = invert(transf[D])
					* translation_Mat4x4f(p0)
					* scaling_Mat4x4f(p7-p0);

				glViewport(0,0,width,height);
				
				// Setup projection
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glOrtho(p0[0],p7[0],p0[1],p7[1],-p7[2],-p0[2]);

				// Set up modelview matrix
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();
				glMultTransposeMatrixf(transf[D].get());

				DepthPeeler peeler(width, height);
				peeler.disable_depth_test2();

				cout << width << " " << height << endl;

 				LDImage* ldi = ldi_set.add_ldi(width, height, imat, 
																			 invert(transf[D]));

				LDILayer* ldi_layer = ldi->add_layer();
	
				glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

				draw_with_normals_as_color();

				glReadPixels(0,0,width,height,
										 GL_DEPTH_COMPONENT, GL_FLOAT, 
										 ldi_layer->get_depth_buffer());
				glReadPixels(0,0,width,height,
										 GL_RGB, GL_FLOAT, 
										 ldi_layer->get_normal_buffer());	
				glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

				draw();

				glReadPixels(0,0,width,height,
										 GL_RGBA, GL_FLOAT, 
										 ldi_layer->get_colour_buffer());	
					
				int q=0;
					
				unsigned int query;
				glGenOcclusionQueriesNV(1,&query);

				while(q<100)
					{
						++q;
						peeler.read_back_depth();
						peeler.enable_depth_test2();
						glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
							
						glBeginOcclusionQueryNV(query);
						draw_with_normals_as_color();
						glEndOcclusionQueryNV();
						unsigned int pixels;
						glGetOcclusionQueryuivNV(query, GL_PIXEL_COUNT_NV, &pixels);
						if (pixels>0) 
							{
 								LDILayer* ldi_layer = ldi->add_layer();
								
 								glReadPixels(0,0,width,height,
 														 GL_DEPTH_COMPONENT, GL_FLOAT, 
 														 ldi_layer->get_depth_buffer());
 								glReadPixels(0,0,width,height,
 														 GL_RGB, GL_FLOAT, 
 														 ldi_layer->get_normal_buffer());	
 								glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
								draw();
 								glReadPixels(0,0,width,height,
 														 GL_RGBA, GL_FLOAT, 
 														 ldi_layer->get_colour_buffer());	
							}
						else
						{
                            break;
						}
						cout << " p = " << pixels << endl;

					}
				glDeleteOcclusionQueriesNV (1, &query);

				cout << " Layers: " << q << endl;
			}
		glViewport(vp[0],vp[1],vp[2],vp[3]);
	}
	
}
