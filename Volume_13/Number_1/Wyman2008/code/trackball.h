/* trackball2.h                   */
/* ------------                   */
/*                                */
/* Interface to trackball2.c file */

/* Set how many trackballs this program will utilize                */
void AllocateTrackballs( int totalNum );

/* set the size of the window--used to compute the correct rotation */
void ResizeTrackballWindow( int width, int height );

/* call when the user first clicks on the screen at location (x, y) */
/*   make sure to specify which trackball you're using              */
void SetTrackballOnClick( int ballNum, int x, int y );

/* call when the user moves the mouse when holding down the mouse   */
/*   button.  The params are the tracball numbeer and the new       */
/*   (x, y) location.                                               */
void UpdateTrackballOnMotion( int ballNum, int x, int y );

/* call to multiply the trackball's matrix onto the GL stack.       */
/*    make sure this is only called in the MODELVIEW stack mode     */
/*    pass in the which trackball you want to use                   */
void MultiplyTrackballMatrix( int ballNum );

/* call to print to standard out the various matrix values.         */
void PrintTrackballMatrix( int ballNum );
void PrintInverseTrackballMatrix( int ballNum );

/* call to multiply the trackball's transpose matrix onto the GL    */
/*    stack.  Make sure this is only called in the MODELVIEW stack  */
/*    mode.  Pass in the which trackball you want to use            */
void MultiplyTransposeTrackballMatrix( int ballNum );

/* call to multiply the trackball's inverse matrix onto the GL      */
/*    stack.  Make sure this is only called in the MODELVIEW stack  */
/*    mode.  Pass in the which trackball you want to use            */
void MultiplyInverseTrackballMatrix( int ballNum );

/* call to multiply the trackball's inverse transpose matrix onto   */
/*    the GL stack.  Make sure this is only called in the MODELVIEW */
/*    stack mode.  Pass in the which trackball you want to use      */
void MultiplyInverseTransposeTrackballMatrix( int ballNum );

/* returns a trackball matrix as a 16-element array                 */
void GetTrackBallMatrix( int ballNum, GLfloat *matrix );

/* sets a trackball to the matrix specified.                        */
void SetTrackballMatrixTo( int ballNum, GLfloat *newMat );