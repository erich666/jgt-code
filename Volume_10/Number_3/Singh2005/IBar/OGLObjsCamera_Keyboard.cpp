#include "StdAfx.H"
#include <FL/Enumerations.H>
#include "OGLObjs_Camera.H"

double OGLObjsCamera::s_dDeltaRot = 2.0 * M_PI / 20.0;
double OGLObjsCamera::s_dDeltaTrans = 0.1;
double OGLObjsCamera::s_dDeltaZoom = 1.1;

bool OGLObjsCamera::HandleKeystroke( const unsigned char in_c, 
                                     const bool in_bShift, 
                                     const bool in_bCntrl )
{
	switch ( in_c )
	   {
       case 'r' : 
           if ( in_bShift == TRUE )
               SpinClockwise( s_dDeltaRot ); 
           else 
               SpinCounterClockwise( s_dDeltaRot ); 
           break;

		case 'x':  
           if ( in_bShift == TRUE )
               PosXAxis(  ); 
           else 
               NegXAxis(  ); 
           break;

		case 'y':  
           if ( in_bShift == TRUE )
               PosYAxis(  ); 
           else 
               NegYAxis(  ); 
           break;

		case 'z':  
           if ( in_bShift == TRUE )
               PosZAxis(  ); 
           else 
               NegZAxis(  ); 
           break;


        case '1':  RotateSelf(0, ((in_bShift == TRUE) ? -1.0 : 1.0) * s_dDeltaRot); break;
        case '2':  RotateSelf(1, ((in_bShift == TRUE) ? -1.0 : 1.0) * s_dDeltaRot); break;
        case '3':  RotateSelf(2, ((in_bShift == TRUE) ? -1.0 : 1.0) * s_dDeltaRot); break;

        case FL_Home: // up arrow
			Reset();
			break;


		case 'h': // up arrow
        case FL_Page_Up :
			if ( in_bShift == TRUE ) {
				PanIn( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaTrans *= 2.0;
			} else {
				SpinClockwise( s_dDeltaRot );
			}
			break;

		case 'l': // up arrow
        case FL_Page_Down :
			if ( in_bShift == TRUE ) {
				PanOut( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaTrans *= 2.0;
			} else {
				SpinCounterClockwise( s_dDeltaRot );
			}
			break;

        case 'i': // up arrow
        case FL_Up :
			if ( in_bShift == TRUE ) {
				PanUp( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaTrans *= 2.0;
			} else {
				RotateUp( s_dDeltaRot );
			}
			break;
		case 'm': // down arrow
        case FL_Down :
			if ( in_bShift == TRUE ) {
				PanDown( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaTrans *= 0.5;
			} else {
				RotateDown( s_dDeltaRot );
			}
			break;
		case 'j': // left arrow
        case FL_Left :
			if ( in_bShift == TRUE ) {
				PanLeft( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaRot *= 0.5;
			} else {
				RotateLeft( s_dDeltaRot );
			}
			break;
		case 'k': // right arrow
        case FL_Right :
			if ( in_bShift == TRUE ) {
				PanRight( s_dDeltaTrans );
			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaRot *= 2.0;
			} else {
				RotateRight( s_dDeltaRot );
			}
			break;
		case 'p' :	// zoom
			if ( in_bShift == TRUE ) {
				PanIn( s_dDeltaTrans );

			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaZoom *= 1.1;
			} else {
				SetZoom( s_dDeltaZoom * GetZoom() );
			}
			break;
		case 'n' :
			if ( in_bShift == TRUE ) {
				PanOut( s_dDeltaZoom );

			} else if ( in_bCntrl == TRUE ) {
				s_dDeltaZoom *= 0.9;
			} else {
				SetZoom( GetZoom() / s_dDeltaZoom );
			}
			break;
		default:
			//TRACE("Unknown char %d\n", Key);
			return FALSE;
	}
    return TRUE;
}
