#ifndef _UTILS_CAMERA_DEFS_H
#define _UTILS_CAMERA_DEFS_H

#include "Rn_Defs.H"
#include "S4_quaternion.H"
#include <GL/gl.h>

using namespace CwMtx;

class OGLObjsVisionCamera;

/** OGLObjsCamera 
*A camera.
*
*<br>   Files: 
*<dd><li> include/OpenGL/OGLObjs_Camera.H
*<dd><li> src/OpenGL/OGLObjs/OGLObjsCamera.cpp */
class OGLObjsCamera  {
protected:
    /// Aspect ratio is the aspect ratio scale (normally 1)
	double m_dNear, m_dFar, m_dZoom, m_dAspectRatio, m_dSkew;
    R2Pt   m_ptCOP; // Center of projection
    double m_dFocusDist; 
	int    m_iWidth, m_iHeight;
    R3Pt   m_ptFocus, m_ptOrthoSize; // At point. m_ptFocus = Eye + look * m_dFocusDist
    
    // rotation stored as a quaternian
    S4Quaternion m_quat;

    // Much of this data is redundent
    // It's stored to avoid recalculating all 4 matrices
    R4Matrix m_matTrans, m_matRot, m_matScl, m_matPersp;
    R4Matrix m_matTransInv, m_matRotInv, m_matSclInv;
    R4Matrix m_matWtoC, m_matCtoW, m_matCtoWTranspose;
    R4Matrix m_matProj, m_matProjNorm, m_matPerspInv;

	bool m_bPerspec;
     
    // For keyboard changes
    static double s_dDeltaRot, s_dDeltaTrans, s_dDeltaZoom;

	void Check() const;
     
    virtual void SetTranslationMatrix( const R3Pt &in_pt );
    virtual void SetRotationMatrix();
    virtual void SetRotationMatrix( const double &in_dAng, const R3Vec &in_veAxis);
    virtual void SetScaleMatrix();
    virtual void SetPerspectiveMatrix();
    virtual void SetFinalMatrices();
    virtual void SetAllMatrices();

public:
	/**@name Access */
    //@{
    ///
    virtual R3Pt  From() const { return m_matTransInv * R3Pt(0,0,0); }
	///
    virtual R3Pt  At()   const { return From() + Look() * m_dFocusDist; } 
    ///
    virtual R3Pt  FocusPoint() const { return m_ptFocus; }
	///
    virtual R3Vec Up()  const { return m_matRotInv * R3Vec(0,1,0); }
	///
    virtual R3Vec Right()  const { return m_matRotInv * R3Vec(1,0,0); }
	///
    virtual R3Vec Look() const { return m_matRotInv * R3Vec(0,0,-1); }
    ///
    R2Pt   GetProjectionCenter() const { return m_ptCOP; }
    ///
    double GetNonUniformScaling() const { return m_dAspectRatio; }
    ///
    double GetProjectionDepth() const { return m_matPersp(2,2); }
	///
    double GetFocusDist() const { return m_dFocusDist; }
	///
    double GetNear() const { return m_dNear; }
	///
    double GetFar() const { return m_dFar; }
	///
    double GetZoom() const { return m_dZoom; }
	///
    double GetAspectRatio() const { return (double) m_iWidth / m_iHeight; }
    ///
    double GetAspectRatioScale() const { return m_dAspectRatio; }
	///
    int    Width() const { return m_iWidth; }
	///
    int    Height() const { return m_iHeight; }
	///
    bool   UsingPerspective() const { return m_bPerspec; }
    ///
    R3Vec  GetCOPOffset() const;
    ///
    const R3Pt &GetFocusPoint() const { return m_ptFocus; }
    //@}

    /**@name Info about camera */
    //@{

    ///The Projection matrix is the product of all the camera matrices.
    ///(i.e. The projection matrix takes world-space points to image-space
    ///points.)
    const R4Matrix &Projection() const { return m_matProj; }
    ///The Perspective matrix takes rotated, translated, scaled points to
    ///image-space points.
    const R4Matrix &PerspectiveMatrix() const { return m_matPersp; }
    ///
    const R4Matrix &CameraToWorld() const { return m_matCtoW; }
    ///The WorldToCamera matrix rotates, translates, and scales
    ///world-space points.
    const R4Matrix &WorldToCamera() const { return m_matWtoC; }
	///
    const R4Matrix &RotationToXYZ() const { return m_matRot; }
	///
    const R4Matrix &RotationFromXYZ() const { return m_matRotInv; }
    ///
    R4Matrix RotTrans() const { return m_matRot * m_matTrans; }
    ///
    const S4Quaternion &GetQuaternion() const { return m_quat; }
	///
    const R4Matrix &Scale() const { return m_matScl; }
    ///
    const R4Matrix &ScaleInv() const { return m_matSclInv; }
    ///
    const R4Matrix &Translation() const { return m_matTrans; }
    ///
    const R4Matrix &InvTranslation() const { return m_matTransInv; }
    //@}

    /**@name Conversion to-from screen/camera coords */
    //@{
	///
    void   GetFourCorners( R3Pt  & out_ptCamera,
                           R3Vec   out_avec[4] ) const;
    /// In 0..width,0..height range. Y is flipped to match 0,0 upper left
    R2Pt     ScreenPt( const R3Pt &in_pt ) const;
    /// In (-1..1, -1..1) range.  
    R2Pt     CameraPt( const R3Pt &in_pt ) const;
    ///
    R3Pt     ProjectedPt( const R3Pt &in_pt ) const;
    ///
    R3Vec    ProjectedNormal( const R3Vec &in_vec ) const;
    ///
    R3Vec    RayFromEye( const R2Pt &in_ptCamera ) const;
    ///
    R3Pt     PtOnScreen( const R2Pt &in_ptCamera ) const;
    ///
    R2Pt     ScreenToCamera( const R2Pt &in_ptScreen ) const;
    ///
    R2Vec    ScreenToCamera( const R2Vec &in_vecScreen ) const;
    /// For use with FlTK event_x() and event_y()
    R2Pt     FlTkToCamera( const int in_iX, const int in_iY ) const;
    /// For use with FlTK event_x() and event_y()
    R2Pt     FlTkToScreen( const int in_iX, const int in_iY ) const;
    ///
    R2Pt     CameraToScreen( const R2Pt &in_ptCamera ) const;
    ///
    R2Vec    CameraToScreen( const R2Vec &in_vecCamera ) const;
	//@}

	/**@name Change camera using key stroke */
	//@{
    ///
    static void SetPanAmount( const double in_d ) { s_dDeltaTrans = in_d; }
    ///
    static void SetRotAmount( const double in_d ) { s_dDeltaRot = in_d; }
    ///
    static void SetZoomAmount( const double in_d ) { s_dDeltaZoom = in_d; }
	///
    bool HandleKeystroke( const unsigned char in_c, const bool in_bShift, const bool in_bCntrl );
    //@}

	/**@name Change camera to look down axes */
	//@{
	///
	void PosXAxis();
	///
	void PosYAxis();
	///
	void PosZAxis();
	///
	void NegXAxis();
	///
	void NegYAxis();
	///
	void NegZAxis();
	//@}

	/**@name Set directly */
	//@{
	///
	void SetNearFar( const double in_dNear, const double in_dFar );
	///
	void SetZoom( const double in_dZoom );
	///
	void SetFromAtUp( const R3Pt &in_ptFrom, const R3Pt &in_ptAt, const R3Vec &in_vecUp );
	///
	void CenterCamera( const R3Pt &in_ptCenter, const R3Pt &in_ptScale );
	///
	void SetSize ( const int in_iWidth, const int in_iHeight );
    ///
    void SetSkew( const double in_dSkew );
    ///
    void SetAspectRatioScale( const double in_d );
    ///
    double GetSkew() const;
    ///
    void SetProjectionCenter( const R2Pt &in_pt );
    ///
    void SetProjectionDepth( const double in_d );
    ///
    void SetQuaternion( const S4Quaternion &in_quat );
    ///
    void SetRotationMatrix( const double in_dRotX, const double in_dRotY, const double in_dRotZ );
    ///
    void SetEye( const R3Pt &in_pt );
    ///
    void SetTranslation( const R3Vec &in_vec );
    ///
    void SetFromOGLObjsVisionCamera( const OGLObjsVisionCamera & );
    ///
    void SetPerspective();
    ///
    void SetOrthogonal( const R3Pt &in_ptScale );
    ///  Makes (-1, -1, 0) and (1,1,0) be the corners of the image
    void SetIdentity();
	///
	void Reset();
	//@}

	/**@name Relative change */
	//@{
	/// Around at point
	void RotateLeft( const double in_dAngle );
	/// Around at point
	void RotateRight( const double in_dAngle );
	/// Around at point
	void RotateUp( const double in_dAngle );
	/// Around at point
	void RotateDown( const double in_dAngle );
	/// Around at point
	void SpinClockwise( const double in_dAngle );
	/// Around at point
	void SpinCounterClockwise( const double in_dAngle );
    /// Rotate around at point. Returns new location of point
    R3Pt RotateAroundPoint( const R3Pt &in_pt, const R4Matrix &in_matRot );
    /// Rotate around at point
    void RotateAroundAt( const int in_iAngle, const double in_dAngle );
    ///
    void RotateAroundFocusPoint( const int in_iAngle, const double in_dAngle );
    /// Virtual trackball
    void RotateVirtualTrackball( const R2Pt &in_ptPrev, const R2Pt &in_ptNext,
                                 const R3Pt &in_ptCenter, const double in_dRadius );

    ///
    void RotateSelf( const int in_iAngle, const double in_dAngle );
    ///
    void AddRotation( const R3Matrix &in_matRot );

	///
	void ZoomIn( const double in_dAmount );
	///
	void ZoomOut( const double in_dAmount );
	///
	void PanLeft( const double in_dAmount );
	///
	void PanRight( const double in_dAmount );
	///
	void PanUp( const double in_dAmount );
	///
	void PanDown( const double in_dAmount );
	///
	void PanIn( const double in_dAmount );
	///
	void PanOut( const double in_dAmount );
    ///
    void Translate( const R3Vec &in_vec );

    /// Also set by CenterCamera
    void SetFocusDistance( const double in_dFocusDist );
    /// 
    virtual void SetFocusPoint( const R3Pt &in_pt, const R3Pt &in_ptScale ) ;
	//@}

	/**@name Set OpenGL's camera */
	//@{
	/// Set the OpenGL viewport and matrices
	virtual void SetOpenGLCamera() const;
    /// Set the GL_PROJECTION and GL_MODELVIEW matrices
	virtual void SetOpenGLCameraMatrices() const;
    /// Set the GL_PROJECTION matrix (with the OGLObjsCamera::PerspectiveMatrix)
    /** The names are confusing, but here's some reference:
      * OGLObjsCamera::Projection() gives the product of the GL_MODELVIEW and GL_PROJECTION matrices.
      * OGLObjsCamera::PerspectiveMatrix() gives the GL_PROJECTION matrix
      * OGLObjsCamera::CameraToWorld() gives the GL_MODELVIEW matrix
      */
	virtual void SetOpenGLProjectionMatrix() const;
    /// Set the GL_MODELVIEW matrix
	virtual void SetOpenGLModelviewMatrix() const;
    ///
    virtual void Draw( const bool in_bIsMouseOver ) const {}
    //@}

    ///
    bool operator==( const OGLObjsCamera & ) const;
	///
	OGLObjsCamera &operator=( const OGLObjsCamera & );
	///
	OGLObjsCamera();
	///
	OGLObjsCamera( const OGLObjsCamera & );
	///
	virtual ~OGLObjsCamera();

	/**@name Read write */
	//@{
	/// 
	virtual void Print( ) const;
	///
	virtual void Read( ifstream & );
	///
	virtual void Write( ofstream & ) const;
	//@}
};

#include "OGLObjs_Camera_i.H"

#endif
