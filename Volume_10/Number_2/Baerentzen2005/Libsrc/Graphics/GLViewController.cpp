#include <fstream>
#ifdef WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include "GLViewController.h"

using namespace std;
using namespace CGLA;

namespace Graphics
{

	GLViewController::GLViewController(int _WINX, int _WINY,
									 const CGLA::Vec3f& _centre, float _rad):
		FOV_DEG(53),
		FOV_RAD((FOV_DEG*M_PI)/180.0f),
		WINX(_WINX), WINY(_WINY), 
		aspect(WINX/WINY),
		centre(_centre), rad(_rad),
		button_down(false),
		spin(false)
	{
		float view_dist = rad/sin(FOV_RAD/2.0f);
		ball = new TrackBall(centre, view_dist, WINX, WINY);
		znear = view_dist - rad;
		zfar  = view_dist + rad;
	
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(FOV_DEG, aspect, znear, zfar);
	}

	void GLViewController::grab_ball(TrackBallAction action, 
																	 const CGLA::Vec2i& pos)
	{
		ball->grab_ball(action,pos);
		if(action==ZOOM_ACTION)
			set_near_and_far();

		spin = false;
		button_down = true;
		last_action = action;
		old_pos     = pos;
	}

	void GLViewController::roll_ball(const CGLA::Vec2i& pos)
	{
		ball->roll_ball(pos);
		if(last_action==ZOOM_ACTION)
			set_near_and_far();
		Vec2f dir = Vec2f(pos-old_pos);
		spin = dir.length()>=1.1f;
		old_pos = pos;	
	}


	void GLViewController::release_ball()
	{
		ball->release_ball();
		if(last_action==ZOOM_ACTION)
			set_near_and_far();
	}

	bool GLViewController::try_spin()
	{
		if(spin && !ball->is_grabbed()) 
			{
				ball->do_spin();
				return true;
			}
		return false;
	}
	
	void GLViewController::set_gl_modelview()
	{
		ball->set_gl_modelview();
	}


	void GLViewController::reshape(int W, int H)
	{
		WINX = W;
		WINY = H;
		aspect = WINX/static_cast<float>(WINY);
		glViewport(0,0,WINX,WINY);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(FOV_DEG, aspect, znear, zfar);
		glMatrixMode(GL_MODELVIEW);
			ball->set_screen_window(WINX, WINY);
	}	

	void GLViewController::set_near_and_far()
	{		
		Vec3f eye, centre, up;
		ball->get_view_param(eye, centre, up);
		float len = (eye-centre).length();
		znear = max(0.01f*rad, len-rad);
		zfar = len+rad;
	
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(FOV_DEG, aspect, znear, zfar);
		glMatrixMode(GL_MODELVIEW);
	}

	bool GLViewController::load(const std::string& s)
	{
		ifstream ifs(s.data(),ifstream::binary);
		if(ifs)
			{
				TrackBall* ball_tmp = ball;
				ifs.read(reinterpret_cast<char*>(this),
								 sizeof(GLViewController));		
				ball = ball_tmp;
				ifs.read(reinterpret_cast<char*>(ball),sizeof(TrackBall));		
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				gluPerspective(FOV_DEG, aspect, znear, zfar);
				glMatrixMode(GL_MODELVIEW);
				ball->set_screen_window(WINX, WINY);
				return true;
			}
		return false;
	}
	bool GLViewController::save(const std::string& s) const
	{
		ofstream ofs(s.data(),ofstream::binary);
		if(ofs)
			{
				ofs.write(reinterpret_cast<const char*>(this),
									sizeof(GLViewController));
				ofs.write(reinterpret_cast<const char*>(ball),sizeof(TrackBall));
				return true;
			}
		return false;
 	}

}
