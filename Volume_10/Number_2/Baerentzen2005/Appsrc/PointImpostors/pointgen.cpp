#include <GLEW/glew.h>
#include <GL/glut.h>
#include <cfloat>
#include "Common/ArgExtracter.h"
#include "CGLA/Mat2x3f.h"
#include "LDI/LDI.h"
#include "LDI/create_ldi.h"

using namespace CGLA;
using namespace std;
using namespace LDI;

namespace 
{
	int ldisize = 300;
	string input_name;
	string output_name = "out.pts";
	float unit_scale;

	Vec3f make_ran_point()
	{
		Vec3f p;
		do
			{
				p = Vec3f(float(rand())/RAND_MAX,
									float(rand())/RAND_MAX,
									float(rand())/RAND_MAX);
			}
		while(p==Vec3f(0,0,0));
		return p;
	}

	bool visible = false;

	void vf(int state)
	{
		if(state=GLUT_VISIBLE)
		{
			visible = true;
		}
		glutPostRedisplay();
	}

	void display_convert()
	{
		if(visible)
			{
				vector<PointRecord> points;

				LDISet ldi_set(ldisize);
				create_ldi(ldi_set);
				ldi_set.convert_to_points(points);
				save_points(output_name, points, ldi_set.unit_scale);

				int N = points.size();
				cout << "Total number of points " << N << endl;
				exit(0);
			}
	}

}

int main(int argc, char**argv)
{
	CMN::ArgExtracter ae(argc, argv);

	ae.extract("-l", ldisize);
	ae.extract("-o", output_name);
	
	if(ae.no_remaining_args() > 0)
		input_name = ae.get_last_arg();
	else
		{
			cout << " No file " << endl;
			exit(1);
		}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH);
	glutInitWindowSize(ldisize, ldisize);
	glutCreateWindow("Point generation");

	int bits;
	glGetIntegerv(GL_DEPTH_BITS, &bits);
	cout << "Depth bits " << bits << endl;

	glutVisibilityFunc(vf);
	glutDisplayFunc(display_convert);

	glewInit();
	glutMainLoop();
}
