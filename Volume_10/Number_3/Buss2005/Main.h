
#include "Node.h"
#include "Tree.h"
#include "Jacobian.h"

#ifndef _MAIN_HEADER
#define _MAIN_HEADER

#pragma comment (lib, "glui32.lib")    /* link with Win32 GLUI lib */


#define RADIAN(X)	((X)*RadiansToDegrees)

const char *WINDOWTITLE = { "Kinematics -- Sam Buss and Jin-Su Kim" };
const char *GLUITITLE   = { "User Interface Window" };

const float ANGFACT = { 1. };
const float SCLFACT = { 0.005f };

#define ROTATE		0
#define SCALE		1

const float MINSCALE = { 0.05f };

const int LEFT   = { 4 };
const int MIDDLE = { 2 };
const int RIGHT  = { 1 };

const int ORTHO = { false };
const int PERSP = { true  };

enum Shape {YSHAPE, DBLYSHAPE, HUMANOID, THROW};
enum Method {JACOB_TRANS, PURE_PSEUDO, DLS, SDLS, COMPARE};

#define RESET		0
#define QUIT		1
#define RUNTEST		2

const float BACKCOLOR[] = { 0., 0., 0., 0. };

const float AXES_COLOR[] = { 1., .5, 0. };
const float AXES_WIDTH   = { 3. };

#define FALSE	0
#define TRUE	1

int	ActiveButton;		/* current button that is down		*/
int	AxesList;		/* list to hold the axes		*/
int	AxesOn;			/* ON or OFF				*/
GLUI *	Glui;			/* instance of glui window		*/
int	GluiWindow;		/* the glut id for the glui window	*/
int	GrWindow;		/* window id for graphics window	*/
int	LeftButton;		/* either ROTATE or SCALE		*/
float	RotMatrix[4][4];	/* set by glui rotation widget		*/
float	Scale, Scale2;		/* scaling factors			*/
int WhichShape;
int WhichMethod;
int	WhichProjection;	/* ORTHO or PERSP			*/
int	Xmouse, Ymouse;		/* mouse values				*/
float	Xrot, Yrot;		/* rotation angles in degrees		*/
float	TransXYZ[3];		/* set by glui translation widgets	*/
int JointLimitsOn;
int RestPositionOn;
int UseJacobianTargets;
int EigenVectorsOn;
int RotAxesOn;

void	Animate( void );
void	Axes( float length );
void	Buttons( int );
void	Display( void );
void	InitGlui( void );
void	InitGraphics( void );
void	InitLists( void );
void	Keyboard(unsigned char, int, int);
void	MouseButton( int, int, int, int );
void	MouseMotion( int, int );
void	resizeWindow( int, int );
void	Reset( void );
void RunTest();
void RunTestA();
void RunTestB();
void RunTestC();
void RunTestD();
void UpdateTargets( double T );
void DoUpdateStep();

#define MAX_NUM_NODE	1000
#define MAX_NUM_THETA	1000
#define MAX_NUM_EFFECT	100

VectorR3 target[MAX_NUM_EFFECT];

Tree treeY;
Jacobian *jacobY;
Node* nodeY[MAX_NUM_NODE];

Tree treeDoubleY;
Jacobian *jacobDoubleY;
Node* nodeDoubleY[MAX_NUM_NODE];

Tree treeDoubleYDLS;
Tree treeDoubleYSDLS;
Jacobian *jacobDoubleYDLS;
Jacobian *jacobDoubleYSDLS;
Node* nodeDoubleYDLS[MAX_NUM_NODE];
Node* nodeDoubleYSDLS[MAX_NUM_NODE];

#endif