// Copyright (C) 2000 Alias|Wavefront, a division of Silicon Graphics Limited.
// 
// The information in this file is provided for the exclusive use of the
// licensees of Alias|Wavefront.  Such users have the right to use, modify,
// and incorporate this code into other products for purposes authorized
// by the Alias|Wavefront license agreement, without fee.
// 
// ALIAS|WAVEFRONT DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
// EVENT SHALL ALIAS|WAVEFRONT BE LIABLE FOR ANY SPECIAL, INDIRECT OR
// CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
// DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
// TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THIS SOFTWARE.

#include <string.h>
#include <iostream.h>
#include <math.h>

#include <maya/MPxLocatorNode.h> 

#include <maya/MFnTypedAttribute.h>

#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnPlugin.h>

#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MString.h>
#include <maya/M3dView.h>
#include <maya/MPointArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MStringArray.h>
#include <maya/MGlobal.h>

#define kPI  3.1415927
#define k2PI 6.2831853

#define kCircleDivisions 20
#define equiv(x,y) (fabs(x-y)<0.0001)

static double indx[2*kCircleDivisions];



// protocol for passing deformation function

typedef void (* deformFunction)(MPxNode *node,
								MPoint &,
								const MMatrix &,
								const MMatrix &,
								bool);
#define kDeformFuncMagicNumber  15128

union funcUnion
{
	deformFunction	addr;
	long data[2];
};

// end protocol


class ibaruist : public MPxLocatorNode
{
public:
						ibaruist();
	virtual				~ibaruist();

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );
	static  void*		creator();
	static  MStatus		initialize();

    static void         deform(MPxNode *node,
							   MPoint &,
							   const MMatrix &,
							   const MMatrix &,
							   bool);

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );


public:
    // deformation function attributes
	static  MObject		child1;
	static  MObject		child2;
	static  MObject		child3;
	static  MObject		function;


	// local node attributes

	static  MObject     angle;  		// ibaruist magnitude
	static  MObject     envelope; 		// ibaruist envelope	
	static  MObject     dropoff;  		// ibaruist dropoff
	static  MObject     ibaruistMatrix; 	// ibaruist center and axis
	static  MObject     world;    		// local or world space deformation


	
	static  MObject     camenv;
	static  MObject     xo;
	static  MObject     yo;
	
	MStringArray vals;

	
		static  MObject     sectionCurveInput; // generated curve

	static  MTypeId		id;

	MDoubleArray values;
	MIntArray deep;
	int cnt;
	double radius;
	MPointArray pete;


private:

	// cache for local attr. for efficient access from def. function 

	bool 		worldSpace;	// whether or not to deform in worldSpace
	double   	env,dr,dr2;    // envelope, dropoff and dropoff-squared
	MMatrix  	mat,matinv;
	MMatrix  	matInit,matInitInv;	
	double   	magnitude;


};

MTypeId     ibaruist::id( 0x80068 );
MObject     ibaruist::child1;     
MObject     ibaruist::child2;     
MObject     ibaruist::child3;     
MObject     ibaruist::function;

// local attr.

MObject     ibaruist::angle;
MObject     ibaruist::envelope;
MObject     ibaruist::dropoff;
MObject		ibaruist::ibaruistMatrix;
MObject     ibaruist::world;

MObject     ibaruist::sectionCurveInput; // generated curve
MObject     ibaruist::camenv;
MObject     ibaruist::xo;
MObject     ibaruist::yo;




  

ibaruist::ibaruist() 
{
	cnt=8;
	radius=20;
	deep.append(1);
	deep.append(4);
	values.append(0.1);
	values.append(0.3);
	values.append(0.5);
	values.append(0.8);
	values.append(0.2);
	values.append(0.9);
	values.append(0.9);
	values.append(0.9);

	//MGlobal::executeCommand("listAttr -ud phoneLocator",vals);

	//for (int i=0;i<vals.length();i++)
	//{
	//	printf("%s\n",vals[i].asChar());
	//}

	//double i=0,inc=(k2PI/(double)kCircleDivisions);
	//int j=0;
	//while  (j<kCircleDivisions) 
	//{
	//	indx[j]=cos(i);	
	//	indx[j+kCircleDivisions]=sin(i);
	//	i+=inc;
	//	j++;
	//}
}
ibaruist::~ibaruist() {}

MStatus ibaruist::compute( const MPlug& plug, MDataBlock& data )
{
	
	MStatus returnStatus;
	if ( plug == camenv || plug==dropoff)
	{	

		MFnNurbsCurve crv=data.inputValue(sectionCurveInput, &returnStatus ).asNurbsCurve();
		MDataHandle handle;
		radius = data.inputValue(envelope, &returnStatus ).asDouble();

		handle = data.outputValue(dropoff, &returnStatus );
		handle.set(1);
		handle.setClean();

		MPointArray cvs;

		crv.getCVs(cvs);
		int cc=crv.numCVs()/3;

		int cnt=cvs.length()/3;
		double tt=cvs[0].y;
		int dk=cvs[0].y;

		MIntArray bm;
		bm.setLength(cnt);

		MDoubleArray vls;
		vls.setLength(cnt);

		for (int j=0;j<cnt;j++)
		{
			bm[j]=0;
			vls[j]=cvs[j].x;
		}
		for (int ll=1;ll<dk;ll++)
			bm[int(cvs[ll+1].y)]=1;

		int reind=0;

		for (ll=dk-1;ll>=0;ll--)
		{
			for (j=0;j<cnt;j++)
			{
				if (bm[j])
					continue;
				vls[j]*= cvs[ll+cnt].y;
			}
			if (ll)
				bm[int(cvs[ll+1].y)]=0;
		}

		MArrayDataHandle mh= data.outputArrayValue(camenv);
		int mc=mh.elementCount();
		if (mc!=cnt)
			printf("this is fooked\n");
		for (int i=0; i<cnt ;i++,mh.next()) // pull out info per camera
		{
			handle = mh.outputValue();
			handle.set(vls[i]);
		}
		//mh.setAllClean();

	}
	if ( plug == function )
	{	
		// function asked for by generic deformer.
		// use this as a setup point for cache of local attr.
		// and any other precomputation.

		MDataHandle handle = data.inputValue(world, &returnStatus );
		worldSpace = handle.asBool();
		
		handle = data.inputValue(envelope, &returnStatus );
		env = handle.asDouble();

		handle = data.inputValue(dropoff, &returnStatus );
		dr = handle.asDouble();
		dr2 = dr*dr;
		
		handle = data.inputValue(angle, &returnStatus );
		magnitude=handle.asDouble();
	
		handle = data.inputValue(ibaruistMatrix, &returnStatus );
		matInit = handle.asMatrix();
		matInitInv = matInit.inverse();
		// setting function address. can also use this point to 
		// to context sensitively switch deformation functions

		handle = data.outputValue(function);
		funcUnion onion;
		onion.addr =  (deformFunction) deform;
		handle.set( kDeformFuncMagicNumber, onion.data[0], onion.data[1] );
		return returnStatus;
	}
 
	return MS::kSuccess;
}

void* ibaruist::creator()
{
	return new ibaruist();
}

MStatus ibaruist::initialize()
{

	MFnNumericAttribute nAttr;

	// deformation function attr initializtion
    child1 = nAttr.create( "fchild1", "f1", MFnNumericData::kLong );
	nAttr.setWritable(false);
    child2 = nAttr.create( "fchild2", "f2", MFnNumericData::kLong );
	nAttr.setWritable(false);
    child3 = nAttr.create( "fchild3", "f3", MFnNumericData::kLong );
	nAttr.setWritable(false);
	function = nAttr.create( "function", "f", child1, child2, child3 );
	nAttr.setDefault( kDeformFuncMagicNumber, 0, 0 );
	nAttr.setStorable(false);

	addAttribute( child1 );
	addAttribute( child2 );
	addAttribute( child3 );
	addAttribute( function );

	// local attr initialization
	

	world=nAttr.create( "world", "wo", MFnNumericData::kBoolean);
	    nAttr.setDefault(true);
	    nAttr.setKeyable(true);

	envelope=nAttr.create( "envelope", "en", MFnNumericData::kDouble );
	    nAttr.setDefault(5.0);
	    nAttr.setKeyable(true);

	dropoff=nAttr.create( "dropoff", "dr", MFnNumericData::kDouble );
	    nAttr.setDefault(1.0);
	    nAttr.setKeyable(true);
	
	angle=nAttr.create( "angle", "ang", MFnNumericData::kDouble );
	    nAttr.setDefault(0.0);
	    nAttr.setKeyable(true);


	camenv=nAttr.create( "weight", "w", MFnNumericData::kDouble );
	   nAttr.setDefault(0.0);
	   nAttr.setKeyable(true);
	   nAttr.setStorable(false);
	   nAttr.setArray(true);

	MFnUnitAttribute uAttr;

	xo = uAttr.create( "cursorX", "cx", MFnUnitAttribute::kDistance, 0.0 );
	uAttr.setStorable(false);
	yo = uAttr.create( "cursorY", "cy", MFnUnitAttribute::kDistance, 0.0 );
	uAttr.setStorable(false);

	MFnMatrixAttribute  mAttr;
	ibaruistMatrix=mAttr.create( "locateMatrix", "lm");
	    mAttr.setStorable(false);
		mAttr.setConnectable(true);


	MStatus statt;
	MFnTypedAttribute  tAttr;

	sectionCurveInput=tAttr.create( "sectionCurveInput", "mi",
								  MFnData::kNurbsCurve,&statt);
	tAttr.setStorable(false);
	tAttr.setConnectable(true);

	addAttribute( sectionCurveInput);

	//  deformation attributes
	addAttribute( angle); 
	addAttribute( world);
	addAttribute( envelope);
	addAttribute( dropoff);
	addAttribute( ibaruistMatrix);
	addAttribute( xo);
	addAttribute( yo);
	addAttribute( camenv);

	attributeAffects( ibaruist::world, ibaruist::function );
	attributeAffects( ibaruist::ibaruistMatrix, ibaruist::function );	
    attributeAffects( ibaruist::envelope, ibaruist::function );
    attributeAffects( ibaruist::dropoff, ibaruist::function );
    attributeAffects( ibaruist::angle, ibaruist::function );
    attributeAffects( ibaruist::sectionCurveInput, ibaruist::camenv );
    attributeAffects( ibaruist::sectionCurveInput, ibaruist::dropoff );

 
	return MS::kSuccess;
}

void ibaruist::deform( MPxNode *node,
					 MPoint &pt,
					 const MMatrix &m,
					 const MMatrix &minv,
					 bool initializationFlag )
//
// Method: deform
//
// Description:   Deform the point with a ibaruist algorithm
//
// Arguments:
//   node : the ibaruist node
//	 pt   : the point to be deformed
//   m    : matrix to transform the point into world space
//   minv : the inverse of m
//	 initializationFlag : true when m is the first point in the
//						  geometry being deformed, else false
//
{
	ibaruist * sq= (ibaruist *)node;
	if (sq==NULL)
		return;

	if (sq->env==0.0 || sq->dr==0.0)
		return;
	
	if (initializationFlag)
	{
		if (sq->worldSpace)
		{
			sq->mat= sq->matInit*minv;
			sq->matinv= m*sq->matInitInv;
		}
		else
		{
			sq->mat= sq->matInit;
			sq->matinv= sq->matInitInv;	
		}
	}

		
	pt*=sq->matinv;

	// ibaruist algorithm
	//
	//
	double percent=sq->env;
	if (sq->dr>0)
	{
		double  dd = pt.x*pt.x+pt.z*pt.z;
		if (dd >= sq->dr2) {
			percent=0;
		}
		else {
			double x = dd/sq->dr2 - 1;
			percent *= x*x;
		}
	}
	
	double ff= sq->magnitude*percent;
	if (ff!=0.0)
	{
		double ff2= 1.0/ff;
		double tt=pt.z*ff;
		pt.z= sin(tt)*(ff2-pt.x);
		pt.x= cos(tt)*(pt.x-ff2)+ff2;
	}

	// end of ibaruist algorithm

	pt *= sq->mat;
}

void ibaruist::draw( M3dView & view, const MDagPath & path, 
					  M3dView::DisplayStyle style,
					  M3dView::DisplayStatus status )
{ 

	MPlug enPlug( thisMObject(), sectionCurveInput );
	MObject mobj;
	MStatus stat=enPlug.getValue ( mobj );
	//if (stat==MS::kSuccess)
		//printf("werd\n");
	MFnNurbsCurve cfn(mobj);

	MPointArray cvs;

	cfn.getCVs(cvs);
	int cc=cfn.numCVs();

	if (cvs.length()!=8)
		printf("problems\n");

    view.beginGL();


	MVector v1=(cvs[4]-cvs[0])*(1.0/3.0);
	MVector v2=(cvs[5]-cvs[0])*(1.0/3.0);
	MVector v3=(cvs[6]-cvs[1])*(1.0/3.0);
	MVector v4=(cvs[7]-cvs[1])*(1.0/3.0);

	glLineWidth(6.0);
	glDisable(GL_DEPTH_TEST);
	// draw spokes
	glBegin(GL_LINES);
			glColor3f(0,0,0.8);
			glVertex3f(cvs[0].x,cvs[0].y,cvs[0].z);
			glVertex3f(cvs[1].x,cvs[1].y,cvs[1].z);

			glVertex3f(cvs[3].x,cvs[3].y,cvs[3].z);
			glVertex3f(cvs[2].x,cvs[2].y,cvs[2].z);

			glVertex3f(cvs[0].x,cvs[0].y,cvs[0].z);
			glVertex3f(cvs[0].x+v1.x,cvs[0].y+v1.y,cvs[0].z+v1.z);

			glVertex3f(cvs[0].x,cvs[0].y,cvs[0].z);
			glVertex3f(cvs[0].x+v2.x,cvs[0].y+v2.y,cvs[0].z+v2.z);


			glVertex3f(cvs[1].x,cvs[1].y,cvs[1].z);
			glVertex3f(cvs[1].x+v3.x,cvs[1].y+v3.y,cvs[1].z+v3.z);

			glVertex3f(cvs[1].x,cvs[1].y,cvs[1].z);
			glVertex3f(cvs[1].x+v4.x,cvs[1].y+v4.y,cvs[1].z+v4.z);

			glColor3f(0.8,0,0.8);


			glVertex3f(cvs[0].x+2*v1.x,cvs[0].y+2*v1.y,cvs[0].z+2*v1.z);
			glVertex3f(cvs[0].x+3*v1.x,cvs[0].y+3*v1.y,cvs[0].z+3*v1.z);


			glVertex3f(cvs[1].x+2*v4.x,cvs[1].y+2*v4.y,cvs[1].z+2*v4.z);
			glVertex3f(cvs[1].x+3*v4.x,cvs[1].y+3*v4.y,cvs[1].z+3*v4.z);

			glColor3f(0,0.8,0.8);

			glVertex3f(cvs[0].x+2*v2.x,cvs[0].y+2*v2.y,cvs[0].z+2*v2.z);
			glVertex3f(cvs[0].x+3*v2.x,cvs[0].y+3*v2.y,cvs[0].z+3*v2.z);

			glVertex3f(cvs[1].x+2*v3.x,cvs[1].y+2*v3.y,cvs[1].z+2*v3.z);
			glVertex3f(cvs[1].x+3*v3.x,cvs[1].y+3*v3.y,cvs[1].z+3*v3.z);

			glColor3f(0.8,0,0);

			glVertex3f(cvs[0].x+v1.x,cvs[0].y+v1.y,cvs[0].z+v1.z);
			glVertex3f(cvs[0].x+2*v1.x,cvs[0].y+2*v1.y,cvs[0].z+2*v1.z);

			glVertex3f(cvs[0].x+v2.x,cvs[0].y+v2.y,cvs[0].z+v2.z);
			glVertex3f(cvs[0].x+2*v2.x,cvs[0].y+2*v2.y,cvs[0].z+2*v2.z);


			glColor3f(0,0.8,0);


			glVertex3f(cvs[1].x+v3.x,cvs[1].y+v3.y,cvs[1].z+v3.z);
			glVertex3f(cvs[1].x+2*v3.x,cvs[1].y+2*v3.y,cvs[1].z+2*v3.z);

			glVertex3f(cvs[1].x+v4.x,cvs[1].y+v4.y,cvs[1].z+v4.z);
			glVertex3f(cvs[1].x+2*v4.x,cvs[1].y+2*v4.y,cvs[1].z+2*v4.z);



	glEnd();
	glEnable(GL_DEPTH_TEST);


	MPlug xPlug( thisMObject(), xo );
	 stat=xPlug.setValue (cvs[0].x );
	MPlug yPlug( thisMObject(), yo );
	 stat=yPlug.setValue (cvs[0].y );

	glLineWidth(1.0);

    view.endGL();


}

// standard initialazation procedures
//

extern "C" MStatus initializePlugin( MObject obj )
{ 
	MFnPlugin plugin( obj, "Alias|Wavefront", "1.0", "Any");
	plugin.registerNode( "ibaruist", ibaruist::id, ibaruist::creator, 
						 ibaruist::initialize, MPxNode::kLocatorNode );

	return MS::kSuccess;
}

extern "C" MStatus uninitializePlugin( MObject obj)
{
	MFnPlugin plugin( obj );
	plugin.deregisterNode( ibaruist::id );

	return MS::kSuccess;
}

/*
//
// DESCRIPTION:
///////////////////////////////////////////////////////
MStatus initializePlugin( MObject obj )
{ 
   const MString UserClassify( "utility/color" );
   MString command( "if( `window -exists createRenderNodeWindow` )  {refreshCreateRenderNodeWindow(\"" );

   MFnPlugin plugin( obj, "Alias|Wavefront - Example", "3.0", "Any");
   plugin.registerNode( "ContrastNode", Contrast::id, 
                         Contrast::creator, Contrast::initialize,
                         MPxNode::kDependNode, &UserClassify );
   command += UserClassify;
   command += "\");}\n";

   MGlobal::executeCommand(command);


   return MS::kSuccess;
}

//
// DESCRIPTION:
///////////////////////////////////////////////////////
MStatus uninitializePlugin( MObject obj )
{
   const MString UserClassify( "utility/color" );
   MString command( "if( `window -exists createRenderNodeWindow` )  {refreshCreateRenderNodeWindow(\"" );

   MFnPlugin plugin( obj );
   plugin.deregisterNode( Contrast::id );

   command += UserClassify;
   command += "\");}\n";

   MGlobal::executeCommand(command);

   return MS::kSuccess;
}
*/
