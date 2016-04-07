// ===========================================================================
//  (C) 1996-98 Vienna University of Technology
// ===========================================================================
//  NAME:       bboxarea
//  TYPE:       c++ code
//  PROJECT:    Bounding Box Area
//  CONTENT:    Computes area of 2D projection of 3D oriented bounding box
//  VERSION:    1.0
// ===========================================================================
//  AUTHORS:    ds      Dieter Schmalstieg
//              ep      Erik Pojar
// ===========================================================================
//  HISTORY:
//
//  19-sep-99 15:23:03  ds      last modification
//  01-dec-98 15:23:03  ep      created
// ===========================================================================

#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtRenderArea.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/elements/SoCacheElement.h>
#include <Inventor/elements/SoViewingMatrixElement.h>
#include <Inventor/elements/SoViewVolumeElement.h>
#include <Inventor/manips/SoTransformerManip.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/SbLinear.h>

//----------------------------------------------------------------------------
// SAMPLE CODE STARTS HERE
//----------------------------------------------------------------------------

// NOTE: This sample program requires OPEN INVENTOR!

//indexlist: this table stores the 64 possible cases of classification of 
//the eyepoint with respect to the 6 defining planes of the bbox (2^6=64)
//only 26 (3^3-1, where 1 is "inside" cube) of these cases are valid.
//the first 6 numbers in each row are the indices of the bbox vertices that
//form the outline of which we want to compute the area (counterclockwise
//ordering), the 7th entry means the number of vertices in the outline.
//there are 6 cases with a single face and and a 4-vertex outline, and
//20 cases with 2 or 3 faces and a 6-vertex outline. a value of 0 indicates
//an invalid case.

const
int indexlist[64][7] = 
{
    {-1,-1,-1,-1,-1,-1,   0}, // 0 inside
    { 0, 4, 7, 3,-1,-1,   4}, // 1 left
    { 1, 2, 6, 5,-1,-1,   4}, // 2 right
    {-1,-1,-1,-1,-1,-1,   0}, // 3 -
    { 0, 1, 5, 4,-1,-1,   4}, // 4 bottom
    { 0, 1, 5, 4, 7, 3,   6}, // 5 bottom, left
    { 0, 1, 2, 6, 5, 4,   6}, // 6 bottom, right
    {-1,-1,-1,-1,-1,-1,   0}, // 7 -
    { 2, 3, 7, 6,-1,-1,   4}, // 8 top
    { 0, 4, 7, 6, 2, 3,   6}, // 9 top, left 
    { 1, 2, 3, 7, 6, 5,   6}, //10 top, right
    {-1,-1,-1,-1,-1,-1,   0}, //11 -
    {-1,-1,-1,-1,-1,-1,   0}, //12 -
    {-1,-1,-1,-1,-1,-1,   0}, //13 -
    {-1,-1,-1,-1,-1,-1,   0}, //14 -
    {-1,-1,-1,-1,-1,-1,   0}, //15 -
    { 0, 3, 2, 1,-1,-1,   4}, //16 front
    { 0, 4, 7, 3, 2, 1,   6}, //17 front, left
    { 0, 3, 2, 6, 5, 1,   6}, //18 front, right
    {-1,-1,-1,-1,-1,-1,   0}, //19 -
    { 0, 3, 2, 1, 5, 4,   6}, //20 front, bottom
    { 1, 5, 4, 7, 3, 2,   6}, //21 front, bottom, left
    { 0, 3, 2, 6, 5, 4,   6}, //22 front, bottom, right
    {-1,-1,-1,-1,-1,-1,   0}, //23 -
    { 0, 3, 7, 6, 2, 1,   6}, //24 front, top
    { 0, 4, 7, 6, 2, 1,   6}, //25 front, top, left
    { 0, 3, 7, 6, 5, 1,   6}, //26 front, top, right
    {-1,-1,-1,-1,-1,-1,   0}, //27 -
    {-1,-1,-1,-1,-1,-1,   0}, //28 -
    {-1,-1,-1,-1,-1,-1,   0}, //29 -
    {-1,-1,-1,-1,-1,-1,   0}, //30 -
    {-1,-1,-1,-1,-1,-1,   0}, //31 -
    { 4, 5, 6, 7,-1,-1,   4}, //32 back
    { 0, 4, 5, 6, 7, 3,   6}, //33 back, left
    { 1, 2, 6, 7, 4, 5,   6}, //34 back, right
    {-1,-1,-1,-1,-1,-1,   0}, //35 -
    { 0, 1, 5, 6, 7, 4,   6}, //36 back, bottom
    { 0, 1, 5, 6, 7, 3,   6}, //37 back, bottom, left
    { 0, 1, 2, 6, 7, 4,   6}, //38 back, bottom, right
    {-1,-1,-1,-1,-1,-1,   0}, //39 -
    { 2, 3, 7, 4, 5, 6,   6}, //40 back, top
    { 0, 4, 5, 6, 2, 3,   6}, //41 back, top, left
    { 1, 2, 3, 7, 4, 5,   6}, //42 back, top, right
    {-1,-1,-1,-1,-1,-1,   0}, //43 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //44 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //45 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //46 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //47 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //48 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //49 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //50 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //51 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //52 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //53 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //54 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //55 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //56 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //57 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //58 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //59 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //60 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //61 invalid
    {-1,-1,-1,-1,-1,-1,   0}, //62 invalid
    {-1,-1,-1,-1,-1,-1,   0}  //63 invalid
};

//----------------------------------------------------------------------------
// calculateBoxArea: computes the screen-projected 2D area of an oriented 3D
// bounding box
//----------------------------------------------------------------------------

float calculateBoxArea(
    const SbVec3f       eye,    //eye point (in bbox object coordinates)
    const SbBox3f&      box,    //3d bbox
    const SbMatrix&     mat,    //free transformation for bbox
    const SbViewVolume& volume) //view volume
{
    SbVec3f min = box.getMin(), max = box.getMax(); //get box corners
    
    //compute 6-bit code to classify eye with respect to the 6 defining planes
    //of the bbox
    int pos = ((eye[0] < min[0]) ?  1 : 0)   // 1 = left
            + ((eye[0] > max[0]) ?  2 : 0)   // 2 = right
            + ((eye[1] < min[1]) ?  4 : 0)   // 4 = bottom
            + ((eye[1] > max[1]) ?  8 : 0)   // 8 = top
            + ((eye[2] < min[2]) ? 16 : 0)   // 16 = front
            + ((eye[2] > max[2]) ? 32 : 0);  // 32 = back

    int num = indexlist[pos][6]; //look up number of vertices in outline
    if (!num) return -1.0;       //zero indicates invalid case, return -1

    SbVec3f vertexBox[8],dst[8],tmp;
    //generate 8 corners of the bbox
    vertexBox[0] = SbVec3f (min[0],min[1],min[2]); //     7+------+6
    vertexBox[1] = SbVec3f (max[0],min[1],min[2]); //     /|     /|
    vertexBox[2] = SbVec3f (max[0],max[1],min[2]); //    / |    / |
    vertexBox[3] = SbVec3f (min[0],max[1],min[2]); //   / 4+---/--+5  
    vertexBox[4] = SbVec3f (min[0],min[1],max[2]); // 3+------+2 /    y   z
    vertexBox[5] = SbVec3f (max[0],min[1],max[2]); //  | /    | /     |  /
    vertexBox[6] = SbVec3f (max[0],max[1],max[2]); //  |/     |/      |/
    vertexBox[7] = SbVec3f (min[0],max[1],max[2]); // 0+------+1      *---x

    float sum = 0; int i;
    for(i=0; i<num; i++) //transform all outline corners into 2D screen space
    {
        mat.multVecMatrix(vertexBox[indexlist[pos][i]],tmp); //orient vertex
        volume.projectToScreen(tmp,dst[i]);                  //project
    }

    sum = (dst[num-1][0] - dst[0][0]) * (dst[num-1][1] + dst[0][1]);
    for (i=0; i<num-1; i++) 
        sum += (dst[i][0] - dst[i+1][0]) * (dst[i][1] + dst[i+1][1]);

    return sum * 0.5; //return computed value corrected by 0.5
}

//----------------------------------------------------------------------------
// SAMPLE CODE ENDS HERE
//----------------------------------------------------------------------------

SoNode* node;
SoPath* path;
SoGetBoundingBoxAction* bbaction = 
    new SoGetBoundingBoxAction(SbViewportRegion());
SoGetMatrixAction* matrixaction = 
    new SoGetMatrixAction(SbViewportRegion());

void callbackFunc(void*, SoAction* action)
{
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        SoState* state = action->getState();

        bbaction->apply(node);
        SbBox3f box = bbaction->getBoundingBox();
        matrixaction->apply(path);
        SbMatrix mat = matrixaction->getMatrix();

        SbMatrix camera = SoViewingMatrixElement::get(state);
        SbMatrix a = mat;
        mat.multRight(camera);
        SbMatrix inv = mat.inverse();

        float res = calculateBoxArea(SbVec3f(inv[3][0],inv[3][1],inv[3][2]),
                                     box,a,SoViewVolumeElement::get(state));
        printf ("Area: %f\n",res);

        // to keep us updated...
        SoCacheElement::invalidate(action->getState());
    }
}

//----------------------------------------------------------------------------

void
main(int , char** argv)
{
    // Initialize Inventor. This returns a main window to use.
    // If unsuccessful, exit.
    Widget myWindow = SoXt::init(argv[0]); // pass the app name
    if (myWindow == NULL) exit(1);

    SoSeparator* root = new SoSeparator;
    SoMaterial* myMaterial = new SoMaterial;
    root->ref();
    root->addChild(new SoDirectionalLight);

    myMaterial->diffuseColor.setValue(1.0, 0.0, 0.0);   // Red
    root->addChild(myMaterial);

    SoTransformerManip* manip = new SoTransformerManip;
    root->addChild(manip);

    SoCube* cube = new SoCube;
    node = cube;
    root->addChild(cube);

    SoCallback* callback = new SoCallback;
    callback->setCallback(callbackFunc);
    root->addChild(callback);
  
    SoXtExaminerViewer* myViewer = new SoXtExaminerViewer(myWindow);
    myViewer->setSceneGraph(root);
    myViewer->setTitle("BBox Size");

    SoSearchAction mySearcher;
    mySearcher.setType (SoCube::getClassTypeId(),FALSE);
    mySearcher.setInterest (SoSearchAction::FIRST);
    mySearcher.apply(root);
    path = mySearcher.getPath();

    SoXt::show(myWindow);  // Display main window
    SoXt::mainLoop();      // Main Inventor event loop
}

//----------------------------------------------------------------------------
//eof

