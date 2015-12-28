#include <iostream>
#include "CGLA/CGLA.h"
#include "TrackBall.h"

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/glu.h>

using namespace std;
using namespace CGLA;

namespace Graphics
{

	TrackBall::TrackBall(const Vec3f& _centre, 
											 float _eye_dist,
											 unsigned _width, 
											 unsigned _height):
		centre(_centre), eye_dist(_eye_dist), scale(0.5*_eye_dist), 
		width(_width), height(_height)
	{
		// This size should really be based on the distance from the center of
		// rotation to the point on the object underneath the mouse.  That
		// point would then track the mouse as closely as possible.  This is a
		// simple example, though, so that is left as an exercise.
		ballsize = 2.0f;
		screen_centre = Vec2i(width/2, height/2);
		qrot = Quaternion(0.0, 0.0, 0.0, 1.0);
		qinc = Quaternion(0.0, 0.0, 0.0, 1.0);
		trans = Vec2f(0.0, 0.0);
	}

	void TrackBall::grab_ball(TrackBallAction act, const Vec2i& v)
	{
		set_position(scalePoint(v));
		current_action = act;
	}

	void TrackBall::roll_ball(const Vec2i& v)
	{
		Vec2f w = scalePoint(v); 
	
		switch (current_action) 
			{
			case ROTATE_ACTION:
				rotate(w);
				break;
			
			case PAN_ACTION:
				pan(w);
				break;
			
			case ZOOM_ACTION:
				zoom(w);
				break;
			}
		last_pos = w;	
	}

	// Call this when the user does a mouse down.  
	// Stop the trackball glide, then remember the mouse
	// down point (for a future rotate, pan or zoom).
	void TrackBall::set_position(const Vec2f& _last_pos) 
	{
		stop_spin();
		last_pos = _last_pos;
	}

	// Rotationaly spin the trackball by the current increment.
	// Use this to implement rotational glide.
	void TrackBall::do_spin() 
	{
		qrot = qrot*qinc;
	}

	// Cease any rotational glide by zeroing the increment.
	void TrackBall::stop_spin() 
	{
		qinc.set(0.0, 0.0, 0.0, 1.0);
	}

	void TrackBall::rotate(const Vec2f& new_v) 
	{
		calcRotation(new_v);
		do_spin();	
	}

	void TrackBall::pan(const Vec2f& new_v) 
	{
		trans += (new_v - last_pos) * Vec2f(scale[0], scale[1]);
	}

	void TrackBall::zoom(const Vec2f& new_v) 
	{
		eye_dist += (new_v[1] - last_pos[1]) * scale[2];
	}

	void TrackBall::calcRotation(const Vec2f& new_pos) 
	{
		// Check for zero rotation
		if (new_pos == last_pos) 
			qinc = Quaternion();
		else
			{
				// Form two vectors based on input points, find rotation axis
				Vec3f p1 = Vec3f(new_pos[0], new_pos[1], projectToSphere(new_pos));
				Vec3f p2 = Vec3f(last_pos[0], last_pos[1], projectToSphere(last_pos));
			
				Vec3f q = cross(p1, p2);		/* axis of rotation from p1 and p2 */
				float L = sqrt(1.0f-dot(q,q) / (dot(p1,p1) * dot(p2,p2)));
			
				q.normalize();				/* q' = axis of rotation */
				q *= sqrt((1 - L)/2);	/* q' = q' * sin(phi) */
			
				qinc.set(q[0],q[1],q[2],sqrt((1 + L)/2));
			}
	}

	// Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
	// if we are away from the center of the sphere.
	float TrackBall::projectToSphere(const Vec2f& v) 
	{
		float d, t, z;

		d = v.length();
  
		// Inside sphere 
		if (d < ballsize * 0.70710678118654752440) {   
			z = sqrt(ballsize*ballsize - d*d);
		}
		// On hyperbola 
		else {           
			t = ballsize / 1.41421356237309504880;
			z = t*t / d;
		}

		return z;
	}

	// Scales integer point to the range [-1, 1]
	Vec2f TrackBall::scalePoint(const Vec2i& v) const
	{
		Vec2f w(v[0],height - v[1]);
		w -= Vec2f(screen_centre);
		w /= Vec2f(width,height);
		w = CGLA::v_min(Vec2f(1.0f), CGLA::v_max(Vec2f(-1), 2*w));
		return w; 
	}

	void TrackBall::get_view_param(Vec3f& eye, Vec3f& _centre, Vec3f& up) const
	{
		up  = qrot.apply(Vec3f(0,1,0));
		Vec3f right = qrot.apply(Vec3f(1,0,0));
		_centre = centre - up * trans[1] - right * trans[0]; 
		eye = qrot.apply(Vec3f(0,0,1)*eye_dist) + _centre;
	}


	// Modify the current gl matrix by the trackball rotation and translation.
	void TrackBall::set_gl_modelview() const
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		Vec3f eye;
		Vec3f _centre;
		Vec3f up;
		get_view_param(eye, _centre, up);
	
		gluLookAt(eye[0], eye[1], eye[2],
							_centre[0], _centre[1], _centre[2], 
							up[0],up[1],up[2]);
	}

	bool TrackBall::is_spinning() const
	{
		static const Quaternion null_quat(0,0,0,1);
		if(!(qinc == null_quat))
			return true;
		return false;
	}
}
