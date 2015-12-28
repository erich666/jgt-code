// Copyright (C) 1997-2001 Alias|Wavefront, a division of Silicon Graphics Limited.
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

// $RCSfile: ibarTool.cpp,v $     $Revision: 1.1 $

////////////////////////////////////////////////////////////////////////
// 
// ibarTool.cc
// 
// Description:
//    Interactive tool for moving objects and components.
//
//    This plug-in will register the following two commands in Maya:
//       ibarToolCmd <x> <y> <z>
//       ibarToolContext
// 
////////////////////////////////////////////////////////////////////////
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <maya/MPxToolCommand.h>
#include <maya/MFnPlugin.h>
#include <maya/MArgList.h>
#include <maya/MGlobal.h>
#include <maya/MItSelectionList.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MDagPath.h>

#include <maya/MFnTransform.h>
#include <maya/MItCurveCV.h>
#include <maya/MItSurfaceCV.h>
#include <maya/MItMeshVertex.h>

#include <maya/MPxSelectionContext.h>
#include <maya/MPxContextCommand.h>
#include <maya/M3dView.h>
#include <maya/MFnCamera.h>

#include <maya/MIntArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MPointArray.h>

#include <maya/MFnNurbsCurve.h>
#include <maya/MFnMesh.h>
#include <maya/MBoundingBox.h>
#include <maya/MItDag.h>
#include <maya/MQuaternion.h>
#include <maya/MMatrix.h>
#include <maya/MFloatMatrix.h>


#define CHECKRESULT(stat,msg)     \
	if ( MS::kSuccess != stat ) { \
		cerr << msg << endl;      \
	}

#define kVectorEpsilon 1.0e-3


#define kPI  3.1415927
#define k2PI 6.2831853

#define kCircleDivisions 20

#define equiv(x,y) (fabs(x-y)<0.001)

#define lindex(x)  ((x+2)%4+4)


static double indx[2*kCircleDivisions];


/////////////////////////////////////////////////////////////
//
// The ibar command
//
// - this is a tool command which can be used in tool
//   contexts or in the MEL command window.
//
/////////////////////////////////////////////////////////////
#define		ibarNAME	"ibarToolCmd"
#define		DOIT		0
#define		UNDOIT		1
#define		REDOIT		2

class ibarCmd : public MPxToolCommand
{
public:
	ibarCmd();
	virtual ~ibarCmd(); 

	MStatus     doIt( const MArgList& args );
	MStatus     redoIt();
	MStatus     undoIt();
	bool        isUndoable() const;
	MStatus		finalize();

	MDoubleArray cvalues;
	MIntArray cindices;
	int ns;

public:
	static void* creator();

	void		setVector( double x, double y, double z);
private:
	MVector 	delta;	// the delta vectors
	MStatus 	action( int flag );	// do the work here


};

ibarCmd::ibarCmd( )
{
	setCommandString( ibarNAME );
}

ibarCmd::~ibarCmd()
{}

void* ibarCmd::creator()
{
	return new ibarCmd;
}

bool ibarCmd::isUndoable() const
//
// Description
//     Set this command to be undoable.
//
{
	return true;
}

void ibarCmd::setVector( double x, double y, double z)
{
	delta.x = x;
	delta.y = y;
	delta.z = z;
}

MStatus ibarCmd::finalize()
//
// Description
//     Command is finished, construct a string for the command
//     for journalling.
//
{
    MArgList command;
    command.addArg( commandString() );
    command.addArg( delta.x );
    command.addArg( delta.y );
    command.addArg( delta.z );

	// This call adds the command to the undo queue and sets
	// the journal string for the command.
	//
    return MPxToolCommand::doFinalize( command );
}

MStatus ibarCmd::doIt( const MArgList& args )
//
// Description
// 		Test MItSelectionList class
//
{
	MStatus stat;
	MVector	vector( 1.0, 0.0, 0.0 );	// default delta
	unsigned i = 0;

	switch ( args.length() )	 // set arguments to vector
	{
		case 1:
			vector.x = args.asDouble( 0, &stat );
			break;
		case 2:
			vector.x = args.asDouble( 0, &stat );
			vector.y = args.asDouble( 1, &stat );
			break;
		case 3:
			vector = args.asVector(i,3);
			break;
		case 0:
		default:
			break;
	}
	delta = vector;

	return action( DOIT );
}

MStatus ibarCmd::undoIt( )
//
// Description
// 		Undo last delta translation
//
{
	return action( UNDOIT );
}

MStatus ibarCmd::redoIt( )
//
// Description
// 		Redo last delta translation
//
{
	return action( REDOIT );
}

MStatus ibarCmd::action( int flag )
//
// Description
// 		Do the actual work here to ibar the objects	by vector
//
{
	MStatus stat;
	MVector vector = delta;

	switch( flag )
	{
		case 1:		// undo
			vector.x = -vector.x;
			vector.y = -vector.y;
			vector.z = -vector.z;
			break;
		case 2:		// redo
			break;
		case 0:		// do command
			break;
		default:
			break;
	}

	// Create a selection list iterator
	//

	MSelectionList slist;
 	MGlobal::getActiveSelectionList( slist );
	MItSelectionList iter( slist, MFn::kInvalid, &stat );

	if ( MS::kSuccess == stat ) {
		MDagPath 	mdagPath;		// Item dag path
		MObject 	mComponent;		// Current component
		MSpace::Space spc = MSpace::kWorld;

		// Translate all selected objects
		//
		for ( ; !iter.isDone(); iter.next() ) 
		{
			// Get path and possibly a component
			//
			iter.getDagPath( mdagPath, mComponent );

			MFnTransform transFn( mdagPath, &stat );
			if ( MS::kSuccess == stat ) {
				stat = transFn.translateBy( vector, spc );
				CHECKRESULT(stat,"Error doing translate on transform");
				continue;
			}

			MItCurveCV cvFn( mdagPath, mComponent, &stat );
			if ( MS::kSuccess == stat ) {
				for ( ; !cvFn.isDone(); cvFn.next() ) {
					stat = cvFn.translateBy( vector, spc );
					CHECKRESULT(stat,"Error setting CV");
				}
				cvFn.updateCurve();
			}

			MItSurfaceCV sCvFn( mdagPath, mComponent, true, &stat );
			if ( MS::kSuccess == stat ) {
				for ( ; !sCvFn.isDone(); sCvFn.nextRow() ) {
					for ( ; !sCvFn.isRowDone(); sCvFn.next() ) {
						stat = sCvFn.translateBy( vector, spc );
						CHECKRESULT(stat,"Error setting CV");
					}
				}
				sCvFn.updateSurface();
			}

			MItMeshVertex vtxFn( mdagPath, mComponent, &stat );
			if ( MS::kSuccess == stat ) {
				for ( ; !vtxFn.isDone(); vtxFn.next() ) {
					stat = vtxFn.translateBy( vector, spc );
					CHECKRESULT(stat,"Error setting Vertex");
				}
				vtxFn.updateSurface();
			}
		} // for
	}
	else {
		cerr << "Error creating selection list iterator" << endl;
	}
	return MS::kSuccess;
}


/////////////////////////////////////////////////////////////
//
// The ibarTool Context
//
// - tool contexts are custom event handlers. The selection
//   context class defaults to maya's selection mode and
//   allows you to override press/drag/release events.
//
/////////////////////////////////////////////////////////////
#define     ibarHELPSTR        "drag to ibar selected object"
#define     ibarTITLESTR       "ibarTool"
#define		TOP			0
#define		FRONT		1
#define		SIDE		2
#define		PERSP		3

class ibarContext : public MPxSelectionContext
{
public:
    ibarContext();
    virtual void    toolOnSetup( MEvent & event );
    virtual MStatus doPress( MEvent & event );
    virtual MStatus doDrag( MEvent & event );
    virtual MStatus doRelease( MEvent & event );
    virtual MStatus doEnterRegion( MEvent & event );

	bool bary(double x0,double y0,
			double x1,double y1,
			double x2,double y2,
			double x3,double y3,
			double *w1,double *w2,
			double *nx=NULL,double *ny=NULL);

private:

	
	void updateCrot();
	void initibar();
	void vtow(short ,short , MPoint &,MVector &);

	int currWin;
	MEvent::MouseButtonType downButton;
	M3dView view;
	short startPos_x, endPos_x, start_x, last_x;
	short startPos_y, endPos_y, start_y, last_y;
	ibarCmd * cmd;

	//////////////////////////////////



	
	MDoubleArray values;  // slider values 0..cnt-1




	MIntArray deep;       // has the index from where one branches
	MPointArray centers;  // center of the wheel
	MPointArray rcenters;  // center of the wheel
	MDoubleArray rot;     // rot angle offset
	MDoubleArray env;     // rot angle offset

	MIntArray bm;

	MPointArray sliders2;  // slider start and end vals for branch limbs.
	MIntArray jump;       // has the index from where one branches
	MIntArray pins;       // has the index from where one branches

	MIntArray ringind;
	int ringsize;

	int cnt;      //make a curve with 2*cnt cvs.

	double radius;
	MPoint cc;
	MVector projenv;
	int engage;
	int state; // 0 nothing, 1 slider 2 wheel

	double px,pz;
	double offx,offz;

	MFnNurbsCurve cfn;
	bool cvon;
	bool refresh;


	MPointArray ring;

	MPointArray oring;

	MVectorArray rv; // down stem 6 vectors.
	MDoubleArray lv; // lengths

	MPoint origin,o;
	MVector n;

	MPoint cp,vp;
	double scalep;
	MPoint ctr;


	MVector ovec,nvec;
	MQuaternion res,lrot;

	MMatrix mm,tt;
	MPoint eye;
	int pww,phh;
	int committed;
	int limb;
	MPoint lctr;
	double lproj;
	double nproj;
	double focus,focusl;
	double ole;
	double tproj;
};


ibarContext::ibarContext()
{
	MString str( ibarTITLESTR );
    setTitleString( str );

	// Tell the context which XPM to use so the tool can properly
	// be a candidate for the 6th position on the mini-bar.
	setImage("moveTool.xpm", MPxContext::kImage1 );
	
	cnt=8;
	radius=20;
	offx=offz=0;	
	cvon=false;

}



void ibarContext::initibar()
{
	view = M3dView::active3dView();


	MDagPath camera;
	MStatus stat = view.getCamera( camera );
	if ( stat != MS::kSuccess ) {
		cerr << "Error: M3dView::getCamera" << endl;
		return;
	}
	MFnCamera fnCamera( camera );
	//MVector upDir = fnCamera.upDirection( spc );
	//MVector rightDir = fnCamera.rightDirection( spc );

	focus=fnCamera.centerOfInterest();
	focusl=fnCamera.focalLength();


	int ww=view.portWidth();
	int hh=view.portHeight();



	view.viewToWorld(ww/2,hh/2,origin,n);

	o=origin + focus*n;

//	printf("wid ht %d %d %f",ww,hh,coi);

	MPoint p1;
	MVector vv;
	double coi;

	view.viewToWorld(ww/2,hh/4,p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[0]=p1+coi*vv;

	view.viewToWorld(ww/2,3*hh/4,p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[1]=p1+coi*vv;

	rv[0]=ring[1]-ring[0];
	lv[0]=rv[0].length();

//	printf("pud %f %f %f %f\n",rv[0].x,rv[0].y,rv[0].z,lv[0]);


	view.viewToWorld(ww/2-ww/8,hh/2,p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[2]=p1+coi*vv;

	view.viewToWorld(ww/2+ww/8,hh/2,p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[3]=p1+coi*vv;

	rv[1]=ring[3]-ring[2];
	lv[1]=rv[1].length();


	rv[0]*= (1.0/lv[0]);
	rv[1]*= (1.0/lv[1]);

	/////////////

	// this is correct
	// ---------------
	//rv[2]= o - focus*rv[1] - ring[0];  // o -focus* rv[1] = vanishing pt for a 45 deg line
	//rv[2].normalize();

	double root2 = 1.0/sqrt(2.0);

	MPoint cubvtx= ring[0] + n*(lv[0]*root2) - rv[1]*(lv[0]*root2);

	ring[4]=  origin + (cubvtx-origin)*(focus/((cubvtx-origin)*n));

	MVector test=ring[4]-ring[0];
	lv[2]=test.length();	
	rv[2]=test/lv[2];

	cubvtx= ring[0] + n*(lv[0]*root2) + rv[1]*(lv[0]*root2);

	ring[5]=  origin + (cubvtx-origin)*(focus/((cubvtx-origin)*n));

	test=ring[5]-ring[0];
	lv[3]=test.length();
	rv[3]=test/lv[3];

	cubvtx= ring[1] + n*(lv[0]*root2) + rv[1]*(lv[0]*root2);

	ring[6]=  origin + (cubvtx-origin)*(focus/((cubvtx-origin)*n));

	test=ring[6]-ring[1];
	lv[4]=test.length();
	rv[4]=test/lv[4];

	cubvtx= ring[1] + n*(lv[0]*root2) - rv[1]*(lv[0]*root2);

	ring[7]=  origin + (cubvtx-origin)*(focus/((cubvtx-origin)*n));

	test=ring[7]-ring[1];
	lv[5]=test.length();
	rv[5]=test/lv[5];


	////////////
/*	view.viewToWorld(ww/2 -ww/4,hh/2 - hh/6,  p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[4]=p1+coi*vv;

	view.viewToWorld(ww/2 +ww/4,hh/2 - hh/6,  p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[5]=p1+coi*vv;

	rv[2]=ring[4]-ring[0];
	lv[2]=rv[2].length();

	rv[3]=ring[5]-ring[0];
	lv[3]=rv[3].length();


	view.viewToWorld(ww/2 -ww/4,hh/2 + hh/6,  p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[7]=p1+coi*vv;

	view.viewToWorld(ww/2 +ww/4,hh/2 + hh/6,  p1,vv);
	coi= ((o-p1)*n)/(vv*n);
	ring[6]=p1+coi*vv;	

	rv[4]=ring[6]-ring[1];
	lv[4]=rv[4].length();

	rv[5]=ring[7]-ring[1];
	lv[5]=rv[5].length();


	rv[2]*= (1.0/lv[2]);
	rv[3]*= (1.0/lv[3]);
	rv[4]*= (1.0/lv[4]);
	rv[5]*= (1.0/lv[5]);
*/

	for (int i=0;i<8;i++)
		oring[i]=ring[i];

//	printf("dirvec %f %f %f, %f",rv[2].x,rv[2].y,rv[2].z,lv[2]);
//	printf("dirvec2 %f %f %f, %f",rv[3].x,rv[3].y,rv[3].z,lv[3]);


}




void ibarContext::toolOnSetup( MEvent & )
{
	MString str( ibarHELPSTR );
    setHelpString( str );
		
	cnt=8;
	radius=20;
	offx=offz=0;	
	cvon=false;


	view = M3dView::active3dView();


	MDagPath camera;
	MStatus stat = view.getCamera( camera );
	if ( stat != MS::kSuccess ) {
		cerr << "Error: M3dView::getCamera" << endl;
		return;
	}
	MFnCamera fnCamera( camera );
	//MVector upDir = fnCamera.upDirection( spc );
	//MVector rightDir = fnCamera.rightDirection( spc );

	double coi=fnCamera.centerOfInterest();


	int ww=view.portWidth();
	int hh=view.portHeight();

	ring.setLength(8);
	oring.setLength(8);

	rv.setLength(6);
	lv.setLength(6);


	initibar();
	state=0;


	cvon=false;
	MItDag dagit(MItDag::kBreadthFirst,MFn::kNurbsCurve);
	for (;!dagit.isDone() && !cvon ;dagit.next())
	{
		MObject mobj = dagit.item();
		MFnDagNode dagnode(mobj);
		//if (dagnode.name()=="ibarcurve")
		{
			if ( MS::kSuccess == cfn.setObject(mobj)) 
			{	
				cvon=true;
				cfn.setCVs(ring);
				cfn.updateCurve();
				view.refresh( true);
				refresh=false;

			}
		}
	}
}

MStatus ibarContext::doPress( MEvent & event )
{

	
	MStatus stat;
	refresh=false;

	bool ctrly=event.modifiers()==MEvent::controlKey;

	// process the click for ibar

	event.getPosition( startPos_x, startPos_y );
	view = M3dView::active3dView();

	endPos_x=startPos_x;
	endPos_y=startPos_y;


	MDagPath camera;
	stat = view.getCamera( camera );
	if ( stat != MS::kSuccess ) {
		cerr << "Error: M3dView::getCamera" << endl;
		return stat;
	}
	MFnCamera fnCamera( camera );


	int ww=pww=view.portWidth();
	int hh=phh=view.portHeight();
	
	double coi=fnCamera.centerOfInterest();

	double thresh=0.05*lv[1];
	
   MVector vec;
   MPoint stp;
   view.viewToWorld( startPos_x, startPos_y, stp, vec );


	camera.pop(1);
	MFnTransform tfn(camera);

   	MFloatMatrix mr=fnCamera.projectionMatrix();
	for (int io=0;io<4;io++)
		for (int jo=0;jo<4;jo++)
			mm[io][jo]=mr[io][jo];
	tt=tfn.transformationMatrix().inverse();
	//MMatrix tt2= tt*mm;
	//MMatrix minv=tt.inverse();

	MPoint chk;
	MPoint chk2;
	double xx= startPos_x*2.0/double(ww) -1;	
	double yy= startPos_y*2.0/double(hh) -1;	
	double zz=-3.0;
	chk2=MPoint(xx,yy,zz);
	chk2*=mm.inverse();
	chk2.z*=-1;
	chk2*=tt.inverse();
	chk2.cartesianize();
	eye=fnCamera.eyePoint(MSpace::kWorld);
	MVector ox=chk-chk2;
	ox.normalize();


	coi= ((o-stp)*n)/(vec*n);
   
   vp=cp=stp+coi*vec;

   if (ctrly)
   {
		MPoint tmp;
		MVector tmp2;
		vtow( startPos_x, startPos_y, tmp, tmp2 );
		vp=tmp+coi*tmp2;
   }

//	printf("%f %f %f %f\n",rv[0].x,rv[0].y,rv[0].z,lv[0]);
   double dt=((cp-ring[0])*rv[0]);

   double dt2=((cp-ring[0])*rv[1]);

   bool onstem= (dt>0 && dt <lv[0]) && 	fabs(dt2)<thresh;


	if (fabs(dt-0.5*lv[0])<thresh && onstem) //translate
	{
		state=1;	

	} 
	else 
	{
		dt=((cp-ring[2])*rv[1]);
		dt2=((cp-ring[2])*rv[0]);
		
		bool oncross= (dt>0 && dt <lv[1]) && 	fabs(dt2)<thresh;

		if ( (fabs(dt)<thresh || fabs(dt-lv[1])<thresh) && fabs(dt2)<thresh ) // scale
		{
			ctr= ring[2]*0.5+ring[3]*0.5;
			state=2;	
		}
		else if (onstem && oncross) 
		{
			state=1;
		}
		else if (onstem || oncross) // rotate
		{
			ovec=cp-o;
			ovec.normalize();
			state=3;	
		}
	}
	MPoint proj;
	


	dt=((cp-ring[0])*rv[2]);

	if (dt>0 && dt < lv[2])
	{
		proj= ring[0] + dt*rv[2];
		double d2= (cp-proj)*(cp-proj);
		if (fabs(d2)<thresh)
		{
			state= 4 +  int(3.0*dt/double(lv[2]));
			limb=2;
			lctr=ring[0];
			lproj=dt;
		}
	}

	dt=((cp-ring[0])*rv[3]);

	if (dt>0 && dt < lv[3])
	{
		proj= ring[0] + dt*rv[3];
		double d2= (cp-proj)*(cp-proj);
		if (fabs(d2)<thresh)
		{
			state= 4 +  int(3.0*dt/double(lv[3]));
			if (state==6)
				state=7;
			limb=3;
			lctr=ring[0];
			lproj=dt;
		}
	}

	dt=((cp-ring[1])*rv[4]);

	if (dt>0 && dt < lv[4])
	{
		proj= ring[1] + dt*rv[4];
		double d2= (cp-proj)*(cp-proj);
		if (fabs(d2)<thresh)
		{
			state= 4 +  int(3.0*dt/double(lv[4]));
			if (state==5)
				state=8;
			if (state==6)
				state=7;
			limb=4;
			lctr=ring[1];
			lproj=dt;
		}
	}

	dt=((cp-ring[1])*rv[5]);

	if (dt>0 && dt < lv[5])
	{
		proj= ring[1] + dt*rv[5];
		double d2= (cp-proj)*(cp-proj);
		if (fabs(d2)<thresh)
		{
			state= 4 +  int(3.0*dt/double(lv[5]));
			if (state==5)
				state=8;
			limb=5;
			lctr=ring[1];
			lproj=dt;
		}
	}

	committed=0;
	ole=focus;
	tproj=lproj;

//	printf("state = %d\n",state);

	if (state==0)	
	{
		stat= MPxSelectionContext::doPress( event );

		if (!isSelecting() ) {

 			MBoundingBox box;
			box.clear();
			bool sally=false;

			if ( MS::kSuccess == stat ) {
				MDagPath 	mdagPath;		// Item dag path
				MObject 	mComponent;		// Current component
				MSpace::Space spc = MSpace::kWorld;

				// Translate all selected objects
				//



				/*
				for ( ; !iter.isDone(); iter.next() ) 
				{
				// Get path and possibly a component
				//
					iter.getDagPath( mdagPath, mComponent );

					MFnDagNode transFn( mdagPath, &stat );
					MBoundingBox bb=transFn.boundingBox();
					box.expand(bb);
					sally=true;
					cvon=true;
				} // for*/
			}  
		}
	}
 
	return stat;
}

MStatus ibarContext::doDrag( MEvent & event )
{
	MStatus stat;

	bool ctrly=event.modifiers()==MEvent::controlKey;

	
	if (state==0)
		stat = MPxSelectionContext::doDrag( event );

	// If we are not in selecting mode (i.e. an object has been selected)
	// then do the translation.
	//
	//if ( !isSelecting() ) 
	else
	{
		last_x=endPos_x;
		last_y=endPos_y;

		event.getPosition( endPos_x, endPos_y );
		MPoint endW, startW,lw;
		MVector vec,lvec;
		view.viewToWorld( startPos_x, startPos_y, startW, vec );
		view.viewToWorld( endPos_x, endPos_y, endW, vec );
		view.viewToWorld( last_x, last_y, lw, lvec );

		if (ctrly && state!=1)
		{
			vtow(startPos_x, startPos_y, startW, vec );
			vtow(endPos_x, endPos_y, endW, vec );
		}

		MDagPath camera;
		stat = view.getCamera( camera );
		if ( stat != MS::kSuccess ) {
			cerr << "Error: M3dView::getCamera" << endl;
			return stat;
		}
		MFnCamera fnCamera( camera );


		int ww=view.portWidth();
		int hh=view.portHeight();
		
		double coi=fnCamera.centerOfInterest();

		camera.pop(1);
		MFnTransform tfn(camera);
		
		//downButton = event.mouseButton();

		coi= ((o-endW)*n)/(vec*n);


		MPoint ovp=vp;
		

		vp=endW+ coi*vec;



		if (state==1)
		{
			//camera.pop(1);
			//MFnTransform tfn(camera);
				
			//MVector ress=tfn.translation(MSpace::kWorld);
			//stat=tfn.translateBy(vp-ovp,MSpace::kWorld);
			if (ctrly)
			{
				ovp=lw+coi*lvec;
				stat=tfn.translateBy(ovp-vp,MSpace::kWorld);
			}
			else
			{
				for (int i = 0;i<8;i++)
					ring[i]+=(vp-ovp);

				cfn.setCVs(ring);
				cfn.updateCurve();
			}

			view.refresh( true);
			refresh=false;
		}
		else if (state==2)
		{
			MVector vt= vp-ctr;
			double scalep= 2.0*fabs((vt*rv[1]))/lv[1];
			vt= ovp-ctr;
			double oscalep= 2.0*fabs((vt*rv[1]))/lv[1];


			if (ctrly)
			{
					double fl=fnCamera.focalLength();
					fnCamera.setFocalLength(fl*scalep/oscalep);

			}
			else
			{
				for (int i = 0;i<8;i++)
				ring[i]= ctr+ (oring[i]-ctr)*scalep;

				cfn.setCVs(ring);
				cfn.updateCurve();
			}	
			view.refresh( true);
			refresh=false;
		}
		else if (state==3)
		{

			MVector nvec= vp-o;

			nvec.normalize();
			res=MQuaternion(ovec,nvec);
			MMatrix mm=res.asMatrix();

			if (ctrly)
			{
				ovec=ovp-o;
				ovec.normalize();
				res=MQuaternion(nvec,ovec);
				stat=tfn.rotateBy(res,MSpace::kWorld);
			}
			else
			{


				for (int i = 0;i<8;i++)
					ring[i]= o+ (oring[i]-o)*mm;

				cfn.setCVs(ring);
				cfn.updateCurve();
			}
			view.refresh( true);
			refresh=false;

		}
		else
		{
			if (!committed)
			{
				if (abs(abs(endPos_x-startPos_x)-abs(endPos_y-startPos_y))>2)
				{
					committed = (abs(endPos_x-startPos_x)>abs(endPos_y-startPos_y));
					if (!committed)
						committed=2; // rotating comm = 1 reducing, increasing length
				}
			}
			if (committed==1)
			{
				nproj= (vp-lctr)*rv[limb];
				if (state==4)
				{
					if (ctrly)
					{
						double scale=nproj/tproj; 
						fnCamera.setAspectRatio(fnCamera.aspectRatio()*scale);
						tproj=nproj;
					}
					ring[4]= ring[0] + rv[2]*(lv[2]*nproj/lproj);
					ring[5]= ring[0] + rv[3]*(lv[3]*nproj/lproj);
					ring[6]= ring[1] + rv[4]*(lv[4]*nproj/lproj);
					ring[7]= ring[1] + rv[5]*(lv[5]*nproj/lproj);
				}
				else if (state==5)
				{
					double scale=nproj/tproj;  // scale=0 is rotation +x 90 deg. scale =1 no rotation. scale = 2 90 deg other way.
					double ang=(scale-1.0)*kPI/2.0;
					if (limb<4)
						ang=-ang;
					MQuaternion rotx(-ang,rv[1]);
					tfn.setRotatePivot(o,MSpace::kWorld,true);
					stat=tfn.rotateBy(rotx,MSpace::kTransform);
					tproj=nproj;

					ring[4]= ring[0] + rv[2]*(lv[2]*nproj/lproj);
					ring[5]= ring[0] + rv[3]*(lv[3]*nproj/lproj);
				}

				else if (state==6)
				{
					double scale=nproj/tproj;  // scale=0 is rotation +x 90 deg. scale =1 no rotation. scale = 2 90 deg other way.

					double ang=(scale-1.0)*kPI/2.0;
					if (limb==2 || limb==5)
						ang=-ang;
					MQuaternion rotx(ang,rv[0]);
					//MVector rp=;
					tfn.setRotatePivot(o,MSpace::kWorld,true);
					stat=tfn.rotateBy(rotx,MSpace::kTransform);
					tproj=nproj;	

					ring[4]= ring[0] + rv[2]*(lv[2]*nproj/lproj);
					ring[7]= ring[1] + rv[5]*(lv[5]*nproj/lproj);
				}
				else if (state==7)
				{
					ring[5]= ring[0] + rv[3]*(lv[3]*nproj/lproj);
					ring[6]= ring[1] + rv[4]*(lv[4]*nproj/lproj);
				}
				else if (state==8)
				{
					ring[6]= ring[1] + rv[4]*(lv[4]*nproj/lproj);
					ring[7]= ring[1] + rv[5]*(lv[5]*nproj/lproj);
				}

				cfn.setCVs(ring);
				cfn.updateCurve();
				view.refresh( true);
				refresh=false;

			}
			else if (committed==2)
			{
				MVector proj= (vp-lctr);
				proj.normalize();
				lrot=MQuaternion(rv[limb],proj);
				MMatrix mm[2];
				mm[0]=lrot.asMatrix();
				mm[1]=mm[0].inverse();

				MVector oproj= (ovp-lctr);
				oproj.normalize();
				MQuaternion olrot=MQuaternion(rv[limb],oproj);				
				
				//if (!ctrly)
					ring[lindex(limb)]=   oring[(lindex(limb))/2-2] + (oring[lindex(limb)]    -oring[(lindex(limb))/2-2])*mm[0];

				if (state==4)
				{
					if (ctrly)
					{

						MVector axis;
						double theta;
						lrot.getAxisAngle(axis,theta);
						axis=vp-lctr;
						axis.normalize();
						double distance = axis*rv[1];
						double le=lv[0]*0.5;

						if (!equiv(distance,1))
						{
							le*=1.0/tan(acos(distance));
							/*if (limb%2)
								theta=-theta;
							tfn.translateBy(axis*theta*8,MSpace::kWorld);
							*/
							tfn.translateBy(n*(ole-le),MSpace::kWorld);
							double fl=fnCamera.focalLength();
							fnCamera.setFocalLength(focusl*le/focus);
							fnCamera.setCenterOfInterest(le);
							ole=le;
						}
					}
					//else
					{

						ring[lindex(limb+1)]= oring[(lindex(limb+1))/2-2] + (oring[lindex(limb+1)]    -oring[(lindex(limb+1))/2-2])*mm[1];
						ring[lindex(limb+2)]= oring[(lindex(limb+2))/2-2] + (oring[lindex(limb+2)]    -oring[(lindex(limb+2))/2-2])*mm[0];
						ring[lindex(limb+3)]= oring[(lindex(limb+3))/2-2] + (oring[lindex(limb+3)]    -oring[(lindex(limb+3))/2-2])*mm[1];
					}
				}
				else if (state==5)
				{
					if (ctrly)
					{
						MVector axis;
						double theta;
						lrot.getAxisAngle(axis,theta);
						if (state==5)
							axis=vp-lctr;
						else
							axis=lctr-vp;

						axis.normalize();
						double distance = axis*rv[1];
						double le=lv[0]*0.5;
						if (!equiv(distance,1))
						{
							le=focus*tan(acos(distance));
							int ww=view.portWidth();
							double vert=fnCamera.verticalFilmAperture();
							int hh=view.portHeight();
							short xx,yy;
							view.worldToView( o+rv[0]*(le-lv[0]*0.5),xx,yy);
							double offy= (double(yy-hh/2)*fnCamera.overscan()*fnCamera.aspectRatio()*vert/(double(ww)));

							if (state==5)
							{
								// no. of horiz. pixels covered is ww/overscan.
								// no. of vert. pixels covered is ww/(ar*overscan)

								tfn.translateBy(rv[0]*(ole + le-lv[0]*0.5),MSpace::kWorld);
								fnCamera.setVerticalFilmOffset(fnCamera.verticalFilmOffset()- offy);
							}
							else
							{
								tfn.translateBy(rv[0]*(lv[0]*0.5-le),MSpace::kWorld);
								fnCamera.setVerticalFilmOffset(fnCamera.verticalFilmOffset()+ offy);
							}
						}

					}
					if (limb==2)
						ring[5]= oring[0] + (oring[5]-oring[0])*mm[1];
					else
						ring[4]= oring[0] + (oring[4]-oring[0])*mm[1];
				}

				else if (state==6)
				{
					if (limb==2)
						ring[7]= oring[1] + (oring[7]-oring[1])*mm[1];
					else
						ring[4]= oring[0] + (oring[4]-oring[0])*mm[1];
				}
				else if (state==7)
				{
					if (limb==3)
						ring[6]= oring[1] + (oring[6]-oring[1])*mm[1];
					else
						ring[5]= oring[0] + (oring[5]-oring[0])*mm[1];
				}
				else if (state==8)
				{
					if (limb==4)
						ring[7]= oring[1] + (oring[7]-oring[1])*mm[1];
					else
						ring[6]= oring[1] + (oring[6]-oring[1])*mm[1];
				}

				cfn.setCVs(ring);
				cfn.updateCurve();
				view.refresh( true);
				refresh=false;

			}
		}


	}
	return stat;
}

void ibarContext::vtow(short sx,short sy, MPoint &chk2,MVector &ox )
{
			double xx= sx*2.0/double(pww) -1;	
			double yy= sy*2.0/double(phh) -1;	
			double zz=-3.0;
			chk2=MPoint(xx,yy,zz);
			chk2*=mm.inverse();
			chk2.z*=-1;
			chk2*=tt.inverse();
			chk2.cartesianize();
			//chk2.x*=-1;
			//chk2.y*=-1;
			ox=chk2-eye;
			if (state==4)
			{
				ox.x*=-1;
				ox.y*=-1;
			}
			ox.normalize();

}


MStatus ibarContext::doRelease( MEvent & event )
{
	MStatus stat = MS::kSuccess;//MPxSelectionContext::doRelease( event );
	//if ( !isSelecting() ) {
	//	event.getPosition( endPos_x, endPos_y );

		// Delete the ibar command if we have ibard less then 2 pixels
		// otherwise call finalize to set up the journal and add the
		// command to the undo queue.
		//

	bool shifty=event.modifiers()==MEvent::shiftKey;
	bool ctrly=event.modifiers()==MEvent::controlKey;
	
	MPoint shpt,shopt;
	double dist=10000;

	if (shifty)
	{
		event.getPosition( endPos_x, endPos_y );
		short xx,yy;
		MPoint nc=(ring[0]+ring[1])*0.5;
		MPoint src;
		MVector dir;
		view.worldToView(nc,xx,yy);
		view.viewToWorld(xx,yy,src,dir);
		MItDag dagit(MItDag::kBreadthFirst,MFn::kMesh);
		MPoint pt;
		for (;!dagit.isDone();dagit.next())
		{
			MObject mobj = dagit.item();
			MFnMesh dagnode(mobj);
			{
				MPointArray pts;
				bool intres=dagnode.intersect(src,dir,pts,10*kMFnMeshPointTolerance,MSpace::kObject,NULL,&stat);
				if (intres)
				{
					double dd=(pts[0]-origin)*(pts[0]-origin);
					if (dd<dist)
					{	dist=dd;pt=pts[0];}
				}
			}
		}
		if (dist<10000)
		{
			double np=(pt-origin)*n;
			shpt=pt;
			shopt=origin + n*np;
		}
	}
			


	if (state>0 && state < 9)
			{				

				MDagPath camera;
				stat = view.getCamera( camera );
				//if (camera.apiType()==MFn::kCameraView)
				//{
				//	printf("wazaaa\n");
				//}

				MString nm=camera.fullPathName();


		//		printf("cam name %s\n",nm.asChar());

				if ( stat != MS::kSuccess ) {
					cerr << "Error: M3dView::getCamera" << endl;
					return stat;
				}
				MFnCamera fnCamera( camera );

				//MDagPath path;
				//fnCamera.getPath(path);
				camera.pop(1);
				MFnTransform tfn(camera);
				

				//MObject cobj=camera.transform();

				//if (cobj.apiType()==MFn::kTransform)
				//{
				//	printf("coolio\n");
				//}



				//MFnTransform tfn(cobj,&stat);

				if (state==1)
				{
						//MVector ress=tfn.translation(MSpace::kWorld);
					    if (!ctrly)
						{
							if (shifty && dist<10000)
							{
								stat=tfn.translateBy(shpt-shopt,MSpace::kWorld);
							}
							else
								stat=tfn.translateBy(vp-cp,MSpace::kWorld);
						}
				}
				else if (state==2)
				{
					//MPoint ctr= ring[2]*0.5+ring[3]*0.5;
					if (!ctrly)
					{
						MVector vt= vp-ctr;
						double scalep= 2.0*fabs((vt*rv[1]))/lv[1];
						double fl=fnCamera.focalLength();
						fnCamera.setFocalLength(fl/scalep);
					}
				}
				else if (state==3)
				{
					if (!ctrly) {
						stat=tfn.rotateBy(res,MSpace::kWorld);
					}
				}
				else if (committed==2 && !ctrly)
				{
					if (state==4 ) // dolly
					{
						MVector axis;
						double theta;
						lrot.getAxisAngle(axis,theta);
						axis=ring[5]-ring[0];
						axis.normalize();
						double distance = axis*rv[1];
						double le=lv[0]*0.5;
						if (!equiv(distance,1))
						{
							le*=1.0/tan(acos(distance));
							/*if (limb%2)
								theta=-theta;
							tfn.translateBy(axis*theta*8,MSpace::kWorld);
							*/
							tfn.translateBy(n*(focus-le),MSpace::kWorld);
							double fl=fnCamera.focalLength();
							fnCamera.setFocalLength(fl*le/focus);
							fnCamera.setCenterOfInterest(le);
							//fnCamera.setFocalLength(fl*(focus+theta*8)/focus);
							//fnCamera.setCenterOfInterest(focus-theta*8);
						}

					}
					if (state==5 || state==8) // pan view center along rv[0]
					{
						/*MVector axis;
						double theta;
						lrot.getAxisAngle(axis,theta);
						if (limb%2)
							theta=-theta;
							*/
						MVector axis;
						double theta;
						lrot.getAxisAngle(axis,theta);
						if (state==5)
							axis=ring[5]-ring[0];
						else
							axis=ring[6]-ring[1];

						axis.normalize();
						double distance = axis*rv[1];
						double le=lv[0]*0.5;
						if (!equiv(distance,1))
						{
							le=focus*tan(acos(distance));
							int ww=view.portWidth();
							double vert=fnCamera.verticalFilmAperture();
							int hh=view.portHeight();
							short xx,yy;
							view.worldToView( o+rv[0]*(le-lv[0]*0.5),xx,yy);
							double offy= (double(yy-hh/2)*fnCamera.overscan()*fnCamera.aspectRatio()*vert/(double(ww)));

							if (state==5)
							{
								// no. of horiz. pixels covered is ww/overscan.
								// no. of vert. pixels covered is ww/(ar*overscan)

								tfn.translateBy(rv[0]*(le-lv[0]*0.5),MSpace::kWorld);
								fnCamera.setVerticalFilmOffset(fnCamera.verticalFilmOffset()- offy);
							}
							else
							{
								tfn.translateBy(rv[0]*(lv[0]*0.5-le),MSpace::kWorld);
								fnCamera.setVerticalFilmOffset(fnCamera.verticalFilmOffset()+ offy);
							}
						}
					}
					if (state==6 || state==7) // pan view ctr along rv[1]
					{
						MVector axis;
						double theta;
						if (state==6)
							axis=ring[7]-ring[1];
						else
							axis=ring[5]-ring[0];

						axis.normalize();
						double distance = axis*rv[1];
						double le=lv[0]*0.5;
						le=focus*tan(acos(distance));
						double vert=fnCamera.horizontalFilmAperture();
						int hh=view.portWidth();
						short xx,yy;
								view.worldToView( o+rv[1]*(le-lv[0]*0.5),xx,yy);
								double offy= (double(xx-hh/2)*fnCamera.overscan()*vert/double(hh));
								tfn.translateBy(rv[1]*(lv[0]*0.5-le),MSpace::kWorld);
								fnCamera.setHorizontalFilmOffset(fnCamera.horizontalFilmOffset()+ offy);

						//MVector axis;
						//double theta;
						//lrot.getAxisAngle(axis,theta);
						//if (limb%2)
						//	theta=-theta;
						//fnCamera.setHorizontalFilmOffset(fnCamera.horizontalFilmOffset()+theta);

					}

				}
				else if (committed==1)
				{
					if (state==4) // change aspect ratio
					{
						double scale=nproj/lproj; 
						fnCamera.setAspectRatio(fnCamera.aspectRatio()*scale);
					}	
					if (/*state==5 ||*/ state==8) // rotate around rv[1]
					{
						double scale=nproj/lproj;  // scale=0 is rotation +x 90 deg. scale =1 no rotation. scale = 2 90 deg other way.
						double ang=(scale-1.0)*kPI/2.0;
						if (limb<4)
							ang=-ang;
						MQuaternion rotx(-ang,rv[1]);
						tfn.setRotatePivot(o,MSpace::kWorld,true);
						stat=tfn.rotateBy(rotx,MSpace::kTransform);

					}
					if (/*state==6 ||*/ state==7) // rotate around rv[0]
					{
						double scale=nproj/lproj;  // scale=0 is rotation +x 90 deg. scale =1 no rotation. scale = 2 90 deg other way.

						double ang=(scale-1.0)*kPI/2.0;
						if (limb==2 || limb==5)
							ang=-ang;
						MQuaternion rotx(ang,rv[0]);
						//MVector rp=;
						tfn.setRotatePivot(o,MSpace::kWorld,true);
						stat=tfn.rotateBy(rotx,MSpace::kTransform);
					}
				}

				state=0;
				view.refresh( true);
				initibar();
				cfn.setCVs(ring);
				cfn.updateCurve();
				view.refresh( true);

				refresh=false;

			}

		if ( abs(startPos_x - endPos_x) < 2 && abs(startPos_y - endPos_y) < 2 ) {
			//delete cmd;
			//view.refresh( true );
		}
		else {
			//stat = cmd->finalize();
			//view.refresh( true );
		}
//	}
		state=0;
	return stat;
}

MStatus ibarContext::doEnterRegion( MEvent & event )
//
// Print the tool description in the help line.
//
{
	MString str( ibarHELPSTR );
    return setHelpString( str );
}


bool ibarContext::bary(double x0,double y0,
			double x1,double y1,
			double x2,double y2,
			double x3,double y3,
			double *w1,double *w2,
			double *nx,double *ny)
{

	double b0 =  (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
	double b1 = ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0)) / b0; 
	double b2 = ((x3 - x0) * (y1 - y0) - (x1 - x0) * (y3 - y0)) / b0;
	double b3 = ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / b0;

	if (nx!=NULL && (b1<0 && b2>=0 && b3>=0))
	{
		b1=b2+b3;
		b2/= b1;
		b3/= b1;
		b1=0;
	}
	if (nx!=NULL)
	{
		*nx= x1*b1+x2*b2+x3*b3;
		*ny= y1*b1+y2*b2+y3*b3;
	}

	*w1=b1;
	*w2=b2;

	return (b1>=0 && b2>=0 && b3>=0);
}

/////////////////////////////////////////////////////////////
//
// Context creation command
//
//  This is the command that will be used to create instances
//  of our context.
//
/////////////////////////////////////////////////////////////
#define     CREATE_CTX_NAME	"ibarToolContext"

class ibarContextCommand : public MPxContextCommand
{
public:
    ibarContextCommand() {};
    virtual MPxContext * makeObj();

public:
    static void* creator();
};

MPxContext * ibarContextCommand::makeObj()
{
    return new ibarContext();
}

void * ibarContextCommand::creator()
{
    return new ibarContextCommand;
}


///////////////////////////////////////////////////////////////////////
//
// The following routines are used to register/unregister
// the commands we are creating within Maya
//
///////////////////////////////////////////////////////////////////////
MStatus initializePlugin( MObject obj )
{
	MStatus		status;
	MFnPlugin	plugin( obj, "Alias|Wavefront - Example", "3.0", "Any" );

	status = plugin.registerContextCommand( CREATE_CTX_NAME,
									&ibarContextCommand::creator,
									ibarNAME, &ibarCmd::creator );
	if (!status) {
		status.perror("registerContextCommand");
		return status;
	}

	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus		status;
	MFnPlugin	plugin( obj );

	status = plugin.deregisterContextCommand( CREATE_CTX_NAME, ibarNAME );
	if (!status) {
		status.perror("deregisterContextCommand");
		return status;
	}

	return status;
}

