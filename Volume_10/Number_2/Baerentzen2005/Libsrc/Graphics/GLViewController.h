#include "TrackBall.h"

namespace Graphics
{

	class GLViewController
	{
		float FOV_DEG;
		float FOV_RAD;
		int WINX, WINY;
		float znear, zfar;
		float aspect;
		CGLA::Vec3f centre;
		float rad;
		bool button_down;
		TrackBallAction last_action;
		CGLA::Vec2i old_pos;
		bool spin;

		TrackBall *ball;
	public:

		GLViewController(int _WINX, int _WINY,
										 const CGLA::Vec3f& _centre, float _rad);
		void grab_ball(TrackBallAction action, const CGLA::Vec2i& pos);
		void roll_ball(const CGLA::Vec2i& pos);
		void release_ball();
		bool try_spin();
		void set_gl_modelview();
		void reshape(int W, int H);
		void set_near_and_far();

		float get_fovy_rad() const {return FOV_RAD;}

		float get_eye_dist() const
		{
			return ball->get_eye_dist();
		}

		void get_view_param(CGLA::Vec3f& e, 
												CGLA::Vec3f& c, CGLA::Vec3f& u) const
		{
			ball->get_view_param(e,c,u);
		}

		bool load(const std::string&);
		bool save(const std::string&) const;
	};
	
}
namespace GFX = Graphics;
