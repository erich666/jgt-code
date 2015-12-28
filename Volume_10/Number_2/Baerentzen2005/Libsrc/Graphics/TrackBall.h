#ifndef _TrackBall_
#define _TrackBall_

#include "CGLA/Vec2i.h"
#include "CGLA/Vec2f.h"
#include "CGLA/Vec3f.h"
#include "CGLA/Vec3Hf.h"
#include "CGLA/Quaternion.h"

namespace Graphics
{

	enum TrackBallAction
		{
			NO_ACTION = 0,
			ROTATE_ACTION,
			PAN_ACTION,
			ZOOM_ACTION
		};

	/** This class represents a virtual tracball. 
			Use it in GLUT, FLTK or other OpenGL programs to allow the user to 
			spin the model being rendered. It needs work to be used with non-GL
			apps since it calls GL API functions. */
	class TrackBall 
	{
		CGLA::Vec3f centre;
		CGLA::Vec2i screen_centre;

		unsigned width, height;
		CGLA::Quaternion	qrot;
		CGLA::Quaternion	qinc;
		CGLA::Vec2f	trans;
		CGLA::Vec3f scale;
		float	ballsize;
		float eye_dist;
		CGLA::Vec2f last_pos;
		TrackBallAction current_action;

		void rotate(const CGLA::Vec2f&);
		void pan(const CGLA::Vec2f&);
		void zoom(const CGLA::Vec2f&);

		void calcRotation(const CGLA::Vec2f&);
		float projectToSphere(const CGLA::Vec2f&);
		CGLA::Vec2f scalePoint(const CGLA::Vec2i&) const;

		void set_position(const CGLA::Vec2f&);

	public:

		/** First constructor argument is the point we look at. 
				The second argument is the distance to eye point.
				The third is the scaling factor
				the last two arguments are the window dimensions. */
		TrackBall(const CGLA::Vec3f&, float, unsigned, unsigned);

		/// Set window dimensions.
		void set_screen_window(unsigned _width, unsigned _height)
		{
			width = _width;
			height = _height;
			screen_centre[0] = static_cast<int>(width/2.0f);
			screen_centre[1] = static_cast<int>(height/2.0f);
		}
	
		/// set the centre point of rotation
		void set_centre(const CGLA::Vec3f& _centre)
		{
			centre = _centre;
		}

		void set_screen_centre(const CGLA::Vec2i& _screen_centre) 
		{
			screen_centre[0] = _screen_centre[0];
			screen_centre[1] = height - _screen_centre[1];
		}

		const CGLA::Quaternion& get_rotation() const 
		{
			return qrot;
		}

		void set_rotation(const CGLA::Quaternion& _qrot)
		{
			qrot = _qrot;
		}

		void set_eye_dist(float _eye_dist)
		{
			eye_dist = _eye_dist;
		}

		float get_eye_dist() const
		{
			return eye_dist;
		}

		/// Call GL to set up viewing. 
		void set_gl_modelview() const;

		/** Spin. used both to spin while button is pressed and if the ball is just
				spinning while program is idling. */
		void do_spin();

		bool is_spinning() const;
	
		/// Zeroes the rotation value - makes everything stop.
		void stop_spin();
	
		/// Call this function to start action when mouse button is pressed 
		void grab_ball(TrackBallAction,const CGLA::Vec2i&);

		/// Call this function to perform action when user drags mouse
		void roll_ball(const CGLA::Vec2i&);

		/// Call this function to stop action when mouse is released.
		void release_ball() 
		{
			current_action = NO_ACTION;
		}

		/// Returns true if the ball is `grabbed' and not released yet.
		bool is_grabbed() const 
		{
			if(current_action == NO_ACTION) 
				return false;
			return true;
		}

		void get_view_param(CGLA::Vec3f& eye, 
												CGLA::Vec3f& _centre, CGLA::Vec3f& up) const;

		TrackBallAction get_current_action() 
		{
			return current_action;
		}

	};

}
namespace GFX = Graphics;

#endif
