#include <assert.h>
#include <vector>
#include <stack>
#include <map>
#include <iostream>
#include <string>

#include <GLEW/glew.h>
#include <GL/glu.h>
#include <GLUI/glui.h>
#include "Common/ArgExtracter.h"
#include "CGLA/Vec3f.h"
#include "CGLA/Vec3uc.h"
#include "CGLA/Mat4x4f.h"
#include "Graphics/GLViewController.h"
#include "Components/Timer.h"
#include "Impostor.h"

using namespace CGLA;
using namespace GFX;
using namespace LDI;
using namespace std;

namespace 
{
	CMP::Timer tim;
	float time_start;
	int frame_count=0;
	int render_mode=0;
	int no_instances=1;
	float geo_dist;

	GLUI *glui;
	Impostor* imp;
	GLViewController* view_ctrl;
	int main_window;
	int WINX=800, WINY=800;
	int LDISIZE = 200;
	int no_points;
	int total_points;
	std::string fname;
	vector<PointRecord> points;
	unsigned int tex;

	float fps;
}

void reshape(int W, int H)
{
	view_ctrl->reshape(W,H);
}


void animate() 
{
	if ( glutGetWindow() != main_window ) 
		glutSetWindow(main_window);  
	
	view_ctrl->try_spin();
   
  glui->sync_live();
	glutPostRedisplay();
}


void mouse(int button, int state, int x, int y) 
{
	Vec2i pos(x,y);
	if (state==GLUT_DOWN) 
		{
			if (button==GLUT_LEFT_BUTTON) 
				view_ctrl->grab_ball(ROTATE_ACTION,pos);
			else if (button==GLUT_MIDDLE_BUTTON) 
				view_ctrl->grab_ball(ZOOM_ACTION,pos);
			else if (button==GLUT_RIGHT_BUTTON) 
				view_ctrl->grab_ball(PAN_ACTION,pos);
			else if (button==3)
				{
					view_ctrl->grab_ball(ZOOM_ACTION,pos);
					view_ctrl->roll_ball(pos+Vec2i(0,-WINY/10));
					view_ctrl->release_ball();
				}
			else if (button==4)
				{
					view_ctrl->grab_ball(ZOOM_ACTION,pos);
					view_ctrl->roll_ball(pos+Vec2i(0,WINY/10));
					view_ctrl->release_ball();
				}
		}
	else if (state==GLUT_UP)
		view_ctrl->release_ball();
}

void motion(int x, int y) 
{
	view_ctrl->roll_ball(Vec2i(x,y));
}

void keyboard(unsigned char key, int x, int y) 
{
	switch(key) 
		{
		case '\033': exit(0); break;
		}
}

float get_fps(float last_frame)
{
	const int NO_FPS_SLOT = 20;
	static float fps_slots[NO_FPS_SLOT];
	static int fps_slot=0;

	fps_slots[fps_slot] = 1.0f/last_frame;
	fps_slot = (fps_slot+1)%NO_FPS_SLOT;

	float frame_rate=0;
	for(int i=0;i<NO_FPS_SLOT;++i)
		frame_rate += fps_slots[i];
	return frame_rate/NO_FPS_SLOT;
}

void display()
{
	float fovy = view_ctrl->get_fovy_rad();
	const float Z0 = WINY/(2.0f*tan(fovy/2.0f));
	static float old_secs;
	static bool was_here = false;
	if(!was_here) 
		{
			tim.start();
			was_here = true;

			glEnable(GL_COLOR_MATERIAL);
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_CULL_FACE);
			glPointSize(1.0f);
		}

	float secs = tim.get_secs();
	fps = get_fps(secs-old_secs);
	old_secs = secs;

	
	glPushMatrix();
	view_ctrl->set_gl_modelview();

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	geo_dist = view_ctrl->get_eye_dist();
	if(render_mode == 0)
		for(int i=0;i<no_instances;++i)
			no_points = imp->draw_points(Z0,geo_dist);
	else if(render_mode == 1)
		for(int i=0;i<no_instances;++i)
			no_points = imp->draw_points(1);
	
	glPopMatrix();
	glutSwapBuffers();
}


int main(int argc, char** argv)
{
	CMN::ArgExtracter ae(argc, argv);

	ae.extract("-l", LDISIZE);
	string mesh_file = "";
	ae.extract("-f", mesh_file);
	if(ae.no_remaining_args() > 0)
		fname = ae.get_last_arg();
	else
		{
			cout << " No file " << endl;
			exit(1);
		}

	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH|GLUT_MULTISAMPLE);
	glutInitWindowSize(WINX, WINY);
	glutInit(&argc, argv);
	main_window = glutCreateWindow("Point Viewer");
	glewInit();

	imp = new Impostor(mesh_file,fname,1, Vec4f(0,0,0,0), Vec3f(0,0,0));
	
	Vec3f c(0.5);
	float r=1;
	imp->get_bsphere(c,r);
	view_ctrl = new GLViewController(WINX,WINY, c, r);

	cout << c << r << endl;

	glEnable(GL_DEPTH_TEST);

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glClearColor(1,1,1,1);

	glui = GLUI_Master.create_glui( "GLUI" );
	glui->set_main_gfx_window( main_window );

	GLUI_Spinner *no_instances_spinner =
    glui->add_spinner( "No instances:",
											 GLUI_SPINNER_INT, &no_instances);
  no_instances_spinner->set_int_limits(0,100);

  GLUI_EditText *geo_dist_out = 
    glui->add_edittext( "Distance:", GLUI_EDITTEXT_FLOAT, &geo_dist );
  geo_dist_out->disable();

  GLUI_EditText *no_points_out = 
    glui->add_edittext( "No. points:", GLUI_EDITTEXT_INT, &no_points );
  no_points_out->disable();

  GLUI_EditText *fps_out = 
    glui->add_edittext( "FPS :", GLUI_EDITTEXT_FLOAT, &fps );
  fps_out->disable();
	GLUI_RadioGroup* rgr = glui->add_radiogroup(&render_mode);
	glui->add_radiobutton_to_group(rgr, "points");
	glui->add_radiobutton_to_group(rgr, "ALL points");
	
	GLUI_Master.set_glutIdleFunc( animate ); 
	glutMainLoop();

	
}

