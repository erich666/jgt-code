
#include <math.h>

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glui.h>

#include "LinearR3.h"
#include "MathMisc.h"
#include "Node.h"

extern int RotAxesOn;

Node::Node(const VectorR3& attach, const VectorR3& v, double size, Purpose purpose, double minTheta, double maxTheta, double restAngle)
{
	Node::freezed = false;
	Node::size = size;
	Node::purpose = purpose;
	seqNumJoint = -1;
	seqNumEffector = -1;
	Node::attach = attach;		// Global attachment point when joints are at zero angle
	r.Set(0.0, 0.0, 0.0);		// r will be updated when this node is inserted into tree
	Node::v = v;				// Rotation axis when joints at zero angles
	theta = 0.0;
	Node::minTheta = minTheta;
	Node::maxTheta = maxTheta;
	Node::restAngle = restAngle;
	left = right = realparent = 0;
}

// Compute the global position of a single node
void Node::ComputeS(void)
{
	Node* y = this->realparent;
	Node* w = this;
	s = r;							// Initialize to local (relative) position
	while ( y ) {
		s.Rotate( y->theta, y->v );
		y = y->realparent;
		w = w->realparent;
		s += w->r;
	}
}

// Compute the global rotation axis of a single node
void Node::ComputeW(void)
{
	Node* y = this->realparent;
	w = v;							// Initialize to local rotation axis
	while (y) {
		w.Rotate(y->theta, y->v);
		y = y->realparent;
	}
}

// Draw the box from the origin to point r.
void Node::DrawBox() const
{

	glPushMatrix();

	/*	if (r.getx()) {
		beta = atan2(r.gety(), sqrt(r.getx()*r.getx()+r.getz()*r.getz()));
		glRotatef(beta*180./M_PI, -r.getz(), 0.0f, r.getx());
	} else if (r.getz()) {
		beta = atan2(r.gety(), r.getz());
		glRotatef(beta*180./M_PI, -1.0f, 0.0f, 0.0f);			
	} else {
		if (r.gety() >= 0) {
			glRotatef(90., -1.0f, 0.0f, 0.0f);
		} else {
			glRotatef(90., 1.0f, 0.0f, 0.0f);
		}
	} */
	
	if ( r.z!=0.0 || r.x!=0.0 ) {
		double alpha = atan2(r.z, r.x);
		glRotatef(alpha*RadiansToDegrees, 0.0f, -1.0f, 0.0f);
	}

	if ( r.y!=0.0 ) {
		double beta = atan2(r.y, sqrt(r.x*r.x+r.z*r.z));
		glRotatef( beta*RadiansToDegrees, 0.0f, 0.0f, 1.0f );
	}

	double length = r.Norm();
	glScalef(length/size, 1.0f, 1.0f);
	glTranslatef(size/2, 0.0f, 0.0f);

	glutSolidCube(size);

	glPopMatrix();
}

void Node::DrawNode(bool isRoot)
{
	if (!isRoot) {
		DrawBox();
	}

	if (RotAxesOn) {
		const double rotAxisLen = 1.3;
		glDisable(GL_LIGHTING);
		glColor3f(1.0f, 1.0f, 0.0f);
		glLineWidth(2.0);
		glBegin(GL_LINES);
		VectorR3 temp = r;
		temp.AddScaled(v,rotAxisLen*size);
		glVertex3f( temp.x, temp.y, temp.z );
		temp.AddScaled(v,-2.0*rotAxisLen*size);
		glVertex3f( temp.x, temp.y, temp.z );
		glEnd();
		glLineWidth(1.0);
		glEnable(GL_LIGHTING);
	}
	glTranslatef(r.x, r.y, r.z);
	glRotatef(theta*RadiansToDegrees, v.x, v.y, v.z);
}

void Node::PrintNode()
{
	cerr << "Attach : (" << attach << ")\n";
	cerr << "r : (" << r << ")\n";
	cerr << "s : (" << s << ")\n";
	cerr << "w : (" << w << ")\n";
	cerr << "realparent : " << realparent->seqNumJoint << "\n";
}

void Node::InitNode()
{
	theta = 0.0;
}