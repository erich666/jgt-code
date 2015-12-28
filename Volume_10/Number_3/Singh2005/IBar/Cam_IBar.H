#ifndef _CAM_IBAR_DEFS_H__
#define _CAM_IBAR_DEFS_H__

#include "OGLObjs_Camera.H"
#include <vector>

class CAMIBar : public OGLObjsCamera
{
protected:
    typedef enum {
        NONE = 0,
        PAN = 1,
        ZOOM = 2,
        ROTATE = 3,
        LEFT = 4,
        RIGHT = 8,
        TOP = 16,
        BOTTOM = 32,
        ASPECT_RATIO = 64,
        SKEW = 128,
        ALL = 256
    } HandleType;

    // Movement up/down or left/right?
    typedef enum {
        UNCOMMITTED = 0,
        SCALE_ONLY = 1,
        ANGLE_ONLY = 2,
        BOTH = 3
    } CommitType;

    // In camera coordinates. Minimum distance to be from IBar
    double m_dClickDistance;
    // Center the camera on an object or the middle of the screen
    bool   m_bCenterObj;

    // Drawing state
    HandleType m_handle;
    CommitType m_commit;
    bool       m_bCameraMode;     // Use bottom limbs for camera mode
    R2Pt m_ptDown;                // Mouse down point in [-1,1] X [-1,1]
    R2Pt m_ptDrag;                // Mouse move point in [-1,1] X [-1,1]
    int  m_iClosest;              // Which cube point was selected

    // Save camera on mouse downfor camera-centric mode
    OGLObjsCamera m_cameraOrig;

    // normally 0.5 (half the height of the screen)
    // If in object mode, can optionally scale IBar to match object
    double m_dCubeHeight, m_dObjCubeHeight;

    // Screen space limb
    R2Vec Limb( const std::vector<R3Pt> &in_aptCube, const R4Matrix &in_mat ) const;
    R2Pt LimbBase( const std::vector<R3Pt> &in_aptClick, const R4Matrix &in_mat ) const;

    // Build a cube at the focus distance. out_dTStem is the horizontal line (0.5 if no rotation/COP)
    std::vector<R3Pt> MakeCube( double &out_dTStem ) const ;

    void MouseMoveObject() ;

public:
    /**@name Access */
    //@{
    ///
    bool CenterObjectMode() const { return m_bCenterObj; }
    //@}

    /**@name Change behaviour */
    //@{
	///
	void ToggleCenterObject();
    ///
    void SetClickDistance( const double in_d ) ;
    /// 
    void SetFocusPoint( const R3Pt &in_pt, const R3Pt &in_ptScale ) ;
    //@}

    /**@name Rendering */
    //@{
    ///
    void Draw( const bool in_bIsMouseOver ) const;
	/// Set OpenGL matrices based on new (object-centric) or mouse down (camera-centric) camera
	void SetOpenGLCamera() const;
    //@}

    /**@name Mouse movement */
    //@{
    ///
    bool IsMouseOver( const R2Pt_i &in_pt, bool &out_bRedraw ) ;
    ///
    bool MouseDown( const R2Pt_i &in_pt, const bool in_bShift ) ;
    ///
    void MouseMove( const R2Pt_i &in_pt, const bool in_bShift ) ;
    ///
    void MouseRelease( const R2Pt_i &in_pt, const bool in_bShift ) ;
	//@}


    ///
    CAMIBar &operator=( const CAMIBar &in_cam );
    ///
    CAMIBar( const CAMIBar &in_cam );
    ///
    CAMIBar();
    ///
    ~CAMIBar() {}
};

#endif