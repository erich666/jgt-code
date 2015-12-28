#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glui.h>

#include "Main.h"

// Make slowdown factor larger to make the simulation take larger, less frequent steps
// Make the constant factor in Tstep larger to make time pass more quickly
//const int SlowdownFactor = 40;
const int SlowdownFactor = 10;		// Make higher to take larger steps less frequently
const int SleepsPerStep=SlowdownFactor;
int SleepCounter=0;
const double Tstep = 0.0005*(double)SlowdownFactor;		// Time step
double T = -Tstep;				// Current time

/*   FOLLOWING BLOCK OF CODE USED FOR MAKING MOVIES
#include "RgbImage.h"
RgbImage theScreenImage;
int DumpCounter = 0;
int DumpCounterStart = 1000;
int DumpCounterEnd = 1600;
*/

void BuildTreeYShape(Node *node[], Tree &tree)
{
	const VectorR3& unitx = VectorR3::UnitX;
	const VectorR3& unity = VectorR3::UnitY;
	const VectorR3& unitz = VectorR3::UnitZ;
	const VectorR3 unit1(sqrt(14.0)/8.0, 1.0/8.0, 7.0/8.0);
	const VectorR3& zero = VectorR3::Zero;

	//node[0] = new Node(VectorR3(0.0f, -0.5f, 0.0f), unit1, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	node[0] = new Node(VectorR3(0.0f, -0.5f, 0.0f), unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRoot(node[0]);

	node[1] = new Node(VectorR3(0.0f, 0.4f, 0.0f), unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[0], node[1]);

	node[2] = new Node(VectorR3(0.0f, 0.4f, 0.0f), unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRightSibling(node[1], node[2]);

	node[3] = new Node(VectorR3(0.5f, 1.0f, 0.0f), unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[1], node[3]);

	node[4] = new Node(VectorR3(-0.5f, 1.0f, 0.0f), unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[2], node[4]);

	node[5] = new Node(VectorR3(0.7f, 1.3f, 0.0f), unit1, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[3], node[5]);

	node[6] = new Node(VectorR3(-0.8f, 1.5f, 0.0f), unit1, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[4], node[6]);

	node[7] = new Node(VectorR3(0.7f, 2.0f, 0.0f), zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[5], node[7]);

	node[8] = new Node(VectorR3(-0.8f, 1.9f, 0.0f), zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[6], node[8]);
}

void BuildTreeDoubleYShape(Node *node[], Tree &tree)
{
	const VectorR3& unitx = VectorR3::UnitX;
	const VectorR3& unity = VectorR3::UnitY;
	const VectorR3& unitz = VectorR3::UnitZ;
	const VectorR3 unit1(sqrt(14.0)/8.0, 1.0/8.0, 7.0/8.0);
	const VectorR3& zero = VectorR3::Zero;
	VectorR3 p0(0.0f, -1.5f, 0.0f);
	VectorR3 p1(0.0f, -1.0f, 0.0f);
	VectorR3 p2(0.0f, -0.5f, 0.0f);
	VectorR3 p3(0.5f*Root2Inv, -0.5+0.5*Root2Inv, 0.0f);
	VectorR3 p4(0.5f*Root2Inv+0.5f*HalfRoot3, -0.5+0.5*Root2Inv+0.5f*0.5, 0.0f);
	VectorR3 p5(0.5f*Root2Inv+1.0f*HalfRoot3, -0.5+0.5*Root2Inv+1.0f*0.5, 0.0f);
	VectorR3 p6(0.5f*Root2Inv+1.5f*HalfRoot3, -0.5+0.5*Root2Inv+1.5f*0.5, 0.0f);
	VectorR3 p7(0.5f*Root2Inv+0.5f*HalfRoot3, -0.5+0.5*Root2Inv+0.5f*HalfRoot3, 0.0f);
	VectorR3 p8(0.5f*Root2Inv+1.0f*HalfRoot3, -0.5+0.5*Root2Inv+1.0f*HalfRoot3, 0.0f);
	VectorR3 p9(0.5f*Root2Inv+1.5f*HalfRoot3, -0.5+0.5*Root2Inv+1.5f*HalfRoot3, 0.0f);
	VectorR3 p10(-0.5f*Root2Inv, -0.5+0.5*Root2Inv, 0.0f);
	VectorR3 p11(-0.5f*Root2Inv-0.5f*HalfRoot3, -0.5+0.5*Root2Inv+0.5f*HalfRoot3, 0.0f);
	VectorR3 p12(-0.5f*Root2Inv-1.0f*HalfRoot3, -0.5+0.5*Root2Inv+1.0f*HalfRoot3, 0.0f);
	VectorR3 p13(-0.5f*Root2Inv-1.5f*HalfRoot3, -0.5+0.5*Root2Inv+1.5f*HalfRoot3, 0.0f);
	VectorR3 p14(-0.5f*Root2Inv-0.5f*HalfRoot3, -0.5+0.5*Root2Inv+0.5f*0.5, 0.0f);
	VectorR3 p15(-0.5f*Root2Inv-1.0f*HalfRoot3, -0.5+0.5*Root2Inv+1.0f*0.5, 0.0f);
	VectorR3 p16(-0.5f*Root2Inv-1.5f*HalfRoot3, -0.5+0.5*Root2Inv+1.5f*0.5, 0.0f);

	node[0] = new Node(p0, unit1, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRoot(node[0]);

	node[1] = new Node(p1, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[0], node[1]);

	node[2] = new Node(p1, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[1], node[2]);

	node[3] = new Node(p2, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[2], node[3]);

	node[4] = new Node(p2, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRightSibling(node[3], node[4]);

	node[5] = new Node(p3, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[3], node[5]);

	node[6] = new Node(p3, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRightSibling(node[5], node[6]);

	node[7] = new Node(p3, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[5], node[7]);

	node[8] = new Node(p4, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[7], node[8]);

	node[9] = new Node(p5, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[8], node[9]);

	node[10] = new Node(p5, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[9], node[10]);

	node[11] = new Node(p6, zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[10], node[11]);

	node[12] = new Node(p3, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[6], node[12]);

	node[13] = new Node(p7, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[12], node[13]);

	node[14] = new Node(p8, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[13], node[14]);

	node[15] = new Node(p8, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[14], node[15]);

	node[16] = new Node(p9, zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[15], node[16]);

	node[17] = new Node(p10, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[4], node[17]);

	node[18] = new Node(p10, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[17], node[18]);

	node[19] = new Node(p10, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertRightSibling(node[17], node[19]);

	node[20] = new Node(p11, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[18], node[20]);

	node[21] = new Node(p12, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[20], node[21]);

	node[22] = new Node(p12, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[21], node[22]);

	node[23] = new Node(p13, zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[22], node[23]);

	node[24] = new Node(p10, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[19], node[24]);

	node[25] = new Node(p14, unitz, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[24], node[25]);

	node[26] = new Node(p15, unitx, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[25], node[26]);

	node[27] = new Node(p15, unity, 0.08, JOINT, RADIAN(-180.), RADIAN(180.), RADIAN(30.));
	tree.InsertLeftChild(node[26], node[27]);

	node[28] = new Node(p16, zero, 0.08, EFFECTOR);
	tree.InsertLeftChild(node[27], node[28]);
}

FILE *fp;

int main( int argc, char *argv[] )
{
	BuildTreeYShape(nodeY, treeY);
	jacobY = new Jacobian(&treeY);

	BuildTreeDoubleYShape(nodeDoubleY, treeDoubleY);
	jacobDoubleY = new Jacobian(&treeDoubleY);

	BuildTreeDoubleYShape(nodeDoubleYDLS, treeDoubleYDLS);
	jacobDoubleYDLS = new Jacobian(&treeDoubleYDLS);

	BuildTreeDoubleYShape(nodeDoubleYSDLS, treeDoubleYSDLS);
	jacobDoubleYSDLS = new Jacobian(&treeDoubleYSDLS);

	/*
	fp = fopen("./temp.txt", "w");
	fprintf(fp, "X = [\n");
	*/

	glutInit( &argc, argv );
	InitGraphics();
	InitLists();
	Reset();
	InitGlui();
	glutMainLoop();

	/*
	fprintf(fp, "]\n");
	fclose(fp);
	*/

	return 0;
}

void Animate( void )
{
	glutSetWindow( GrWindow );
	glutPostRedisplay();
}

void Buttons( int id )
{
	switch( id ) {
	case RESET:
		Reset();
		break;
	case QUIT:
		Glui->close();
		glFinish();
		glutDestroyWindow( GrWindow );
		exit( 0 );
	case RUNTEST:
		RunTest();
		break;
	}
	Glui->sync_live();
}

void DrawTarget(double T)
{
	GLfloat target_ambient_and_diffuse[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	GLfloat mat_ambient_and_diffuse[] = { 0.2f, 0.2f, 0.8f, 1.0f };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, target_ambient_and_diffuse);

	int numTarget = (WhichShape==YSHAPE) ? 2 : 4;

	for (int i=0; i<numTarget; i++) {
		glPushMatrix();
		glTranslatef(target[i].x, target[i].y, target[i].z);
		glutSolidSphere(0.1, 10, 10);
		glPopMatrix();
	}

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_ambient_and_diffuse);
}

int numIteration = 1;
double error = 0.0;
double errorDLS = 0.0;
double errorSDLS = 0.0;
double sumError = 0.0;
double sumErrorDLS = 0.0;
double sumErrorSDLS = 0.0;
int numWinOverall = 0;
int numWinAt0 = 0;
int numWinAt1 = 0;
int numWinAt2 = 0;
int numWinAt3 = 0;
int numWinAt4 = 0;

#ifdef _DYNAMIC
bool initMaxDist = true;
extern double Excess[];
extern double dsnorm[];
#endif

void Display( void )
{

	DoUpdateStep();

	float scale2;		/* real glui scale factor		*/

	glutSetWindow( GrWindow );

	glDrawBuffer( GL_BACK );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	glShadeModel( GL_FLAT );

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	if (WhichProjection == ORTHO)
		glOrtho(-3., 3.,     -1.7, 1.7,     0.1, 1000.);
		//glOrtho(-3., 3.,     -3., 3.,     0.1, 1000.);
	else
		gluPerspective(75., 1.,	0.1, 1000.);
		// gluPerspective(90., 1.,	0.1, 1000.);

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	gluLookAt( -3., 0., 3.,     0., 0., 0.,     0., 1., 0. );

	glTranslatef( TransXYZ[0], TransXYZ[1], -TransXYZ[2] );

	glRotatef( Yrot, 0., 1., 0. );
	glRotatef( Xrot, 1., 0., 0. );
	glMultMatrixf( (const GLfloat *) RotMatrix );

	glScalef( Scale, Scale, Scale );
	scale2 = 1. + Scale2;		/* because glui translation starts at 0. */
	if( scale2 < MINSCALE )
		scale2 = MINSCALE;
	glScalef( scale2, scale2, scale2 );

	if( AxesOn ) {
		glDisable(GL_LIGHTING);
		glCallList( AxesList );
		glEnable(GL_LIGHTING);
	}

	GLUI_Master.set_glutIdleFunc( Animate );

	DrawTarget(T);

	if (WhichMethod != COMPARE) {
		Tree* tree = ( WhichShape==YSHAPE ) ? &treeY : &treeDoubleY;
		tree->Draw();
	} else {
		GLfloat blue[] = { 0.2f, 0.2f, 0.8f, 1.0f };
		GLfloat green[] = { 0.3f, 0.6f, 0.3f, 1.0f };
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green);
		treeDoubleYDLS.Draw();
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue);
		treeDoubleYSDLS.Draw();
	}

	if (EigenVectorsOn) {
		if ( WhichMethod == SDLS || WhichMethod == DLS || WhichMethod == PURE_PSEUDO ) {
			Jacobian* jacob;
			switch ( WhichShape ) {
				case YSHAPE:
					jacob = jacobY;
					break;
				case DBLYSHAPE:
					jacob = jacobDoubleY;
					break;
				default:
					assert ( 0 );
			}
			jacob->DrawEigenVectors();
		}
	}

	glFlush();

	/*   FOLLOWING BLOCK OF CODE USED FOR MAKING MOVIES
	if ( WhichMethod == JACOB_TRANS && WhichShape==DBLYSHAPE && (SleepCounter%SleepsPerStep)==0 ) { 
		if (DumpCounter==0) {
			T = 0.0;		// Set time back to zero
		} 
		if ( DumpCounter >= DumpCounterStart && DumpCounter<DumpCounterEnd ) {
			theScreenImage.LoadFromOpenglBuffer();
			int fileNum = DumpCounter - DumpCounterStart;
			char filename[23];
			sprintf( filename, "JTRANSPOSE/temp%03d.bmp", fileNum );
			theScreenImage.WriteBmpFile( filename );
		}
		DumpCounter++;
	}
	*/

	glutSwapBuffers();
}

void InitGlui(void)
{
	GLUI_Panel *panel;
	GLUI_RadioGroup *group;
	GLUI_Rotation *rot;
	GLUI_Translation *trans, *scale;

	Glui = GLUI_Master.create_glui( (char *) GLUITITLE, 0, 0, 0);

	Glui->add_statictext( (char *) GLUITITLE );
	Glui->add_separator();

	panel = Glui->add_panel("Shape & Motion");
		group = Glui->add_radiogroup_to_panel( panel, &WhichShape, 0, NULL );
			Glui->add_radiobutton_to_group( group, "Y" );
			Glui->add_radiobutton_to_group( group, "Double Y" );

	panel = Glui->add_panel("Method");
		group = Glui->add_radiogroup_to_panel( panel, &WhichMethod, 0, NULL );
			Glui->add_radiobutton_to_group( group, "Jacobian Transpose" );
			Glui->add_radiobutton_to_group( group, "Pure Pseudoinverse" );
			Glui->add_radiobutton_to_group( group, "Damped Least Squares" );
			Glui->add_radiobutton_to_group( group, "Selectively Damped Least Squares" );
			Glui->add_radiobutton_to_group( group, "Compare" );

	panel = Glui->add_panel("Options");
		//Glui->add_checkbox_to_panel( panel, "Joint Limits", &JointLimitsOn, 0, NULL);
		//Glui->add_checkbox_to_panel( panel, "Rest Position", &RestPositionOn, 0, NULL);
		Glui->add_checkbox_to_panel( panel, "Jacobian Targets", &UseJacobianTargets, 0, NULL);

	panel = Glui->add_panel("View");
		Glui->add_checkbox_to_panel( panel, "Eigen Vectors", &EigenVectorsOn, 0, NULL);
		Glui->add_checkbox_to_panel( panel, "Axes", &AxesOn, 0, NULL);
		Glui->add_checkbox_to_panel( panel, "Perspective", &WhichProjection, 0, NULL);
		Glui->add_checkbox_to_panel( panel, "Rotation Axes", &RotAxesOn, 0, NULL);

	panel = Glui->add_panel( "Object Transformation" );
		rot = Glui->add_rotation_to_panel( panel, "Rotation", (float *) RotMatrix, 0, NULL);
		rot->set_spin( 1.0 );
		Glui->add_column_to_panel( panel, FALSE );
		scale = Glui->add_translation_to_panel( panel, "Scale",  GLUI_TRANSLATION_Y , &Scale2, 0, NULL);
		scale->set_speed( 0.01f );
		Glui->add_column_to_panel( panel, FALSE );
		trans = Glui->add_translation_to_panel( panel, "Trans XY", GLUI_TRANSLATION_XY, &TransXYZ[0], 0, NULL);
		trans->set_speed( 0.1f );
		Glui->add_column_to_panel( panel, FALSE );
		trans = Glui->add_translation_to_panel( panel, "Trans Z",  GLUI_TRANSLATION_Z , &TransXYZ[2], 0, NULL);
		trans->set_speed( 0.1f );

	panel = Glui->add_panel( "", FALSE );
	Glui->add_button_to_panel( panel, "Run Test", RUNTEST, (GLUI_Update_CB) Buttons );
	Glui->add_column_to_panel( panel, FALSE );
	Glui->add_button_to_panel( panel, "Reset", RESET, (GLUI_Update_CB) Buttons );
	Glui->add_button_to_panel( panel, "Quit", QUIT, (GLUI_Update_CB) Buttons );
	Glui->set_main_gfx_window( GrWindow );
	GLUI_Master.set_glutIdleFunc( NULL );
}

void InitGraphics( void )
{
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );

	glutInitWindowSize(500, 300);
	glutInitWindowPosition(300, 200);

	GrWindow = glutCreateWindow( WINDOWTITLE );
	glutSetWindowTitle( WINDOWTITLE );

	glClearColor( BACKCOLOR[0], BACKCOLOR[1], BACKCOLOR[2], BACKCOLOR[3] );

	glutSetWindow( GrWindow );
	glutDisplayFunc( Display );
	glutMouseFunc( MouseButton );
	glutMotionFunc( MouseMotion );
	glutKeyboardFunc( Keyboard );
	glutReshapeFunc( resizeWindow );

	GLfloat global_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	GLfloat light0_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat light0_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
	GLfloat light0_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light0_position[] = { 3.0f, 3.0f, 3.0f, 0.0 };

	GLfloat light1_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat light1_diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat light1_specular[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat light1_position[] = { -6.0f, 3.0f, 3.0f, 0.0 };

	GLfloat mat_ambient_and_diffuse[] = { 0.2f, 0.2f, 0.8f, 1.0f };
	GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat mat_shininess[] = { 15.0f };

	// light model
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);

	// light0
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
	
	// light1
	glLightfv(GL_LIGHT1, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light0_specular);
	glLightfv(GL_LIGHT1, GL_POSITION, light0_position);

	// material properties
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_ambient_and_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
}

void InitLists( void )
{
	AxesList = glGenLists( 1 );
	glNewList( AxesList, GL_COMPILE );
		glColor3fv( AXES_COLOR );
		glLineWidth( AXES_WIDTH );
		Axes( 1.5 );
		glLineWidth( 1. );
	glEndList();
}

void resizeWindow ( int w, int h )
{
	// Define the portion of the window used for OpenGL rendering.
	glViewport( 0, 0, w, h );	// View port uses whole window


}

void MouseButton( int button, int state, int x, int y )
{
	int b;			/* LEFT, MIDDLE, or RIGHT		*/

	switch( button ) {
	case GLUT_LEFT_BUTTON:
		b = LEFT;		break;
	case GLUT_MIDDLE_BUTTON:
		b = MIDDLE;		break;
	case GLUT_RIGHT_BUTTON:
		b = RIGHT;		break;
	default:
		b = 0;
		cerr << "Unknown mouse button: " << button << "\n";
	}

	if( state == GLUT_DOWN ) {
		Xmouse = x;
		Ymouse = y;
		ActiveButton |= b;		/* set the proper bit	*/
	} else {
		ActiveButton &= ~b;		/* clear the proper bit	*/
	}
}

void MouseMotion( int x, int y )
{
	int dx, dy;		/* change in mouse coordinates		*/

	dx = x - Xmouse;		/* change in mouse coords	*/
	dy = y - Ymouse;

	if( ActiveButton & LEFT ) {
		switch( LeftButton ) {
		case ROTATE:
			Xrot += ( ANGFACT*dy );
			Yrot += ( ANGFACT*dx );
			break;
		case SCALE:
			Scale += SCLFACT * (float) ( dx - dy );
			if( Scale < MINSCALE )
				Scale = MINSCALE;
			break;
		}
	}

	if( ActiveButton & MIDDLE ) {
		Scale += SCLFACT * (float) ( dx - dy );
		if( Scale < MINSCALE )
			Scale = MINSCALE;
	}

	Xmouse = x;			/* new current position		*/
	Ymouse = y;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}

void Keyboard(unsigned char c, int x, int y)
{
	Glui->sync_live();
	glutSetWindow(GrWindow);
	glutPostRedisplay();
}

void Reset( void )
{
	ActiveButton = 0;
	AxesOn = false;
	LeftButton = ROTATE;
	Scale  = 1.0;
	Scale2 = 0.0;		/* because add 1. to it in Display()	*/
	WhichProjection = ORTHO;
	Xrot = Yrot = 0.;
	TransXYZ[0] = TransXYZ[1] = TransXYZ[2] = 0.;
	WhichShape = YSHAPE;
	WhichMethod = SDLS;
	RotAxesOn = false;
	JointLimitsOn = false;
	RestPositionOn = false;
	UseJacobianTargets = false;
	EigenVectorsOn = false;

	                  RotMatrix[0][1] = RotMatrix[0][2] = RotMatrix[0][3] = 0.;
	RotMatrix[1][0]                   = RotMatrix[1][2] = RotMatrix[1][3] = 0.;
	RotMatrix[2][0] = RotMatrix[2][1]                   = RotMatrix[2][3] = 0.;
	RotMatrix[3][0] = RotMatrix[3][1] = RotMatrix[3][3]                   = 0.;
	RotMatrix[0][0] = RotMatrix[1][1] = RotMatrix[2][2] = RotMatrix[3][3] = 1.;

	treeY.Init();
	treeY.Compute();
	jacobY->Reset();

	treeDoubleY.Init();
	treeDoubleY.Compute();	
	jacobDoubleY->Reset();

	treeDoubleYDLS.Init();
	treeDoubleYDLS.Compute();
	jacobDoubleYDLS->Reset();

	treeDoubleYSDLS.Init();
	treeDoubleYSDLS.Compute();
	jacobDoubleYSDLS->Reset();

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}

// Update target positions

void UpdateTargets( double T ) {
	switch (WhichShape) {
	case YSHAPE:
		target[0].Set(2.0f+1.5*sin(6*T), -0.5+1.7f+0.2*sin(7*T), 0.3f+0.2*sin(8*T));
		target[1].Set(-0.7f+0.4*sin(4*T), -0.5+1.3f+0.3*sin(4*T), -0.2f+0.2*sin(3*T));
		assert( treeY.GetNumEffector() == 2 );
		break;
	case DBLYSHAPE:
		target[0].Set(2.0f+1.5*sin(3*T)*2, -0.5+1.0f+0.2*sin(7*T)*2, 0.3f+0.7*sin(5*T)*2);
		target[1].Set(0.5f+0.4*sin(4*T)*2, -0.5+0.9f+0.3*sin(4*T)*2, -0.2f+1.0*sin(3*T)*2);
		target[2].Set(-0.5f+0.8*sin(6*T)*2, -0.5+1.1f+0.2*sin(7*T)*2, 0.3f+0.5*sin(8*T)*2);
		target[3].Set(-1.6f+0.8*sin(4*T)*2, -0.5+0.8f+0.3*sin(4*T)*2, -0.2f+0.3*sin(3*T)*2);
		assert( treeDoubleY.GetNumEffector() == 4);
		break;
	}
}


// Does a single update (on one kind of tree)
void DoUpdateStep() {
	if ( WhichMethod!=COMPARE ) {
	
		if ( SleepCounter==0 ) {
			T += Tstep;
			UpdateTargets( T );
		} 

		Jacobian *jacob;
		switch ( WhichShape ) {
			case YSHAPE:
				jacob = jacobY;
				break;
			case DBLYSHAPE:
				jacob = jacobDoubleY;
				break;
			default:
				assert ( 0 );
		}

		if ( UseJacobianTargets ) {
			jacob->SetJtargetActive();
		}
		else {
			jacob->SetJendActive();
		}
		jacob->ComputeJacobian();						// Set up Jacobian and deltaS vectors

		// Calculate the change in theta values 
		switch (WhichMethod) {
			case JACOB_TRANS:
				jacob->CalcDeltaThetasTranspose();		// Jacobian transpose method
				break;
			case DLS:
				jacob->CalcDeltaThetasDLS();			// Damped least squares method
				break;
			case PURE_PSEUDO:
				jacob->CalcDeltaThetasPseudoinverse();	// Pure pseudoinverse method
				break;
			case SDLS:
				jacob->CalcDeltaThetasSDLS();			// Selectively damped least squares method
				break;
			default:
				jacob->ZeroDeltaThetas();
				break;
		}

		if ( SleepCounter==0 ) {
			jacob->UpdateThetas();							// Apply the change in the theta values
			jacob->UpdatedSClampValue();
			SleepCounter = SleepsPerStep;
		}
		else { 
			SleepCounter--;
		}
	}
	else { // COMPARE MODE		(only supports double-Y)
		WhichShape = DBLYSHAPE;
		if ( SleepCounter==0 ) {
			T += Tstep;
			UpdateTargets( T );
		} 
		Jacobian *jacob1 = jacobDoubleYDLS;
		Jacobian *jacob2 = jacobDoubleYSDLS;
		if ( UseJacobianTargets ) {
			jacob1->SetJtargetActive();
			jacob2->SetJtargetActive();
		}
		else {
			jacob1->SetJendActive();
			jacob2->SetJendActive();
		}
		jacob1->ComputeJacobian();						// Set up Jacobian and deltaS vectors
		jacob2->ComputeJacobian();						// Set up Jacobian and deltaS vectors
		jacob1->CalcDeltaThetasDLS();
		jacob2->CalcDeltaThetasSDLS();
		if ( SleepCounter==0 ) {
			jacob1->UpdateThetas();							// Apply the change in the theta values
			jacob1->UpdatedSClampValue();					// Not needed for usual DLS method
			jacob2->UpdateThetas();							// Apply the change in the theta values
			jacob2->UpdatedSClampValue();
			SleepCounter = SleepsPerStep;
		}
		else { 
			SleepCounter--;
		}
	}


}

void RunTest () 
{
	RunTestB();
}

void RunTestC() 
{

	Node* nodesA[MAX_NUM_NODE];
	Tree treeA;
	BuildTreeDoubleYShape( nodesA, treeA );
	Jacobian jacobA( &treeA );

	double time = 0.0;
	while ( true ) {
		time += 3.0*(1.0+sqrt(5.0));
		treeA.Init();
		treeA.Compute();
		jacobA.Reset();
		jacobA.SetJendActive();
		switch ( WhichMethod ) {
			case JACOB_TRANS:
				jacobA.SetCurrentMode(JACOB_JacobianTranspose);
				fprintf(stdout,"Testing Jacobian Transpose method.\n");
				break;
			case DLS:
				jacobA.SetCurrentMode(JACOB_DLS);
				jacobA.SetDampingDLS( 1.0 );
				fprintf( stdout, "Testing Damped Least Squares convergence.\n");
				break;
			case SDLS:
				jacobA.SetCurrentMode(JACOB_SDLS);
				fprintf( stdout, "Testing Selectively Damped Least Squares convergence.\n");
				break;
		}
		UpdateTargets(time);
		long i = 0;
		while ( true ) {
			i++;
			jacobA.ComputeJacobian();
			jacobA.CalcDeltaThetas();
			jacobA.UpdateThetas();							// Apply the change in the theta values
			double totalError = jacobA.UpdateErrorArray();
			const VectorRn& err = jacobA.GetErrorArray();
			jacobA.UpdatedSClampValue();		// Only relevant for SDLS or DLS with clamping (not needed for usual DLS)
			fprintf(stdout, "Iteration %2ld: total error = %7lf, (%6lf, %6lf, %6lf, %6lf).\n", i, totalError,
					err[0], err[1], err[2], err[3]);
			char c = fgetc(stdin);
			if ( c=='.' ) {
				fprintf(stdout, "\n");
				break;
			}
			if ( c=='x' ) {
				return;
			}
		}
	}
}

void RunTestB()
{
	WhichShape = DBLYSHAPE;		// Need to do to update the targets correctly
	Node* nodesA[MAX_NUM_NODE];
	Tree treeA;
	BuildTreeDoubleYShape( nodesA, treeA );
	Jacobian jacobA( &treeA );
	treeA.Init();
	treeA.Compute();
	jacobA.Reset();
	jacobA.SetJendActive();
	jacobA.SetCurrentMode(JACOB_DLS);
	//jacobA.SetDampingDLS( 1.0 );

	Node* nodesB[MAX_NUM_NODE];
	Tree treeB;
	BuildTreeDoubleYShape( nodesB, treeB );
	Jacobian jacobB( &treeB );
	treeB.Init();
	treeB.Compute();
	jacobB.Reset();
	jacobB.SetJendActive();
	jacobB.SetCurrentMode(JACOB_SDLS);
	//jacobB.SetDampingDLS( 1.0 );

	double time = 0.0;
	long j;
	for ( j=0; j<400; j++) {
		time += Tstep;
		UpdateTargets(time);
		jacobA.ComputeJacobian();
		jacobA.CalcDeltaThetas();
		jacobA.UpdateThetas();							// Apply the change in the theta values
		jacobA.UpdateErrorArray();
		jacobA.UpdatedSClampValue();		// Only relevant for SDLS or DLS with clamping (not needed for usual DLS)
		jacobB.ComputeJacobian();
		jacobB.CalcDeltaThetas();
		jacobB.UpdateThetas();							// Apply the change in the theta values
		jacobB.UpdateErrorArray();
		jacobB.UpdatedSClampValue();		// Only relevant for SDLS or DLS with clamping (not needed for usual DLS)
	}
	double totalErrorA = 0.0;
	double totalErrorB = 0.0;
	double relErrorA, relErrorB;
	double netRelErrorA = 0.0;
	double netRelErrorB = 0.0;
	long win1[5] = {0,0,0,0,0};
	long win2[5] = {0,0,0,0,0};
	int b1, b2, ties;
	int NumTests = 2000;
	double fNumTestsCent = 0.01*NumTests;
	for ( j=0; j<NumTests; j++) {
		time += Tstep;
		UpdateTargets(time);
		jacobA.ComputeJacobian();
		jacobA.CalcDeltaThetas();
		jacobA.UpdateThetas();							// Apply the change in the theta values
		totalErrorA += jacobA.UpdateErrorArray();
		jacobA.UpdatedSClampValue();		// Only relevant for SDLS or DLS with clamping (not needed for usual DLS)
		jacobB.ComputeJacobian();
		jacobB.CalcDeltaThetas();
		jacobB.UpdateThetas();							// Apply the change in the theta values
		totalErrorB += jacobB.UpdateErrorArray();
		jacobB.UpdatedSClampValue();		// Only relevant for SDLS or DLS with clamping (not needed for usual DLS)
		Jacobian::CompareErrors( jacobA, jacobB, &relErrorA, &relErrorB );
		netRelErrorA += relErrorA;
		netRelErrorB += relErrorB;
		Jacobian::CountErrors( jacobA, jacobB, &b1, &b2, &ties );
		win1[b1]++;
		win2[b2]++;
	}

	fprintf(stdout, "DLS:  Total error = %8lf.\n", totalErrorA );
	fprintf(stdout, "SDLS: Total error = %8lf.\n", totalErrorB );
	fprintf(stdout, "DLS:  Relative error = %8lf.\n", netRelErrorA );
	fprintf(stdout, "SDLS: Relative error = %8lf.\n", netRelErrorB );
	fprintf(stdout, "DLS:  Number wins  %4.1f%% 0's, %4.1f%% 1's, %4.1f%% 2's, %4.1f%% 3s, %4.1f%% 4's\n",
			win1[0]/fNumTestsCent, win1[1]/fNumTestsCent, win1[2]/fNumTestsCent, win1[3]/fNumTestsCent, win1[4]/fNumTestsCent);
	fprintf(stdout, "SDLS: Number wins  %4.1f%% 0's, %4.1f%% 1's, %4.1f%% 2's, %4.1f%% 3s, %4.1f%% 4's\n",
			win2[0]/fNumTestsCent, win2[1]/fNumTestsCent, win2[2]/fNumTestsCent, win2[3]/fNumTestsCent, win2[4]/fNumTestsCent);
}

void RunTestA() 
{
	Node* nodesA[MAX_NUM_NODE];
	Tree treeA;
	//BuildTreeDoubleYShape( nodesA, treeA );
	BuildTreeYShape( nodesA, treeA );
	Jacobian jacobA( &treeA );

	// Loop over different damping factors
	double startDamp = 0.3;
	double stepDamp = 0.01;
	int numDamp = 30;
	for ( long i=0; i<numDamp; i++ ) {
		treeA.Init();
		treeA.Compute();
		jacobA.Reset();
		jacobA.SetJendActive();
		jacobA.SetCurrentMode(JACOB_DLS);
		double thisDamping = startDamp + i*stepDamp;
		jacobA.SetDampingDLS( thisDamping );
		double time = 0.0;
		long j;
		for ( j=0; j<20; j++) {
			time += Tstep;
			UpdateTargets(time);
			jacobA.ComputeJacobian();
			jacobA.CalcDeltaThetas();
			jacobA.UpdateThetas();							// Apply the change in the theta values
			jacobA.UpdateErrorArray();
			jacobA.UpdatedSClampValue();		// Only relevant for SDLS, but no harm down here
		}
		double totalError = 0.0;
		for ( j=0; j<200; j++) {
			time += Tstep;
			UpdateTargets(time);
			jacobA.ComputeJacobian();
			jacobA.CalcDeltaThetas();
			jacobA.UpdateThetas();							// Apply the change in the theta values
			totalError += jacobA.UpdateErrorArray();
			jacobA.UpdatedSClampValue();		// Only relevant for SDLS, but no harm down here
		}
		fprintf(stdout," Damping = %7.4lf: total error = %lf.\n", thisDamping, totalError );

	}
}

