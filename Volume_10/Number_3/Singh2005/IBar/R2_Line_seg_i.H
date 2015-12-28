template <class T>
inline
WINbool R2Line_segTC<T>::IsPtOnSeg(const R2PointTC<T> &in_pt) const
{
    R2PointTC<T> out_pt;
    T out_t, out_d;
    return FindPtOnSeg(in_pt, out_pt, out_t, out_d);
}

template <class T>
inline
T R2Line_segTC<T>::Dist_to_seg(const R2PointTC<T> &in_pt) const
{
    R2PointTC<T> out_pt;
    T out_t, out_d;
    FindPtOnSeg(in_pt, out_pt, out_t, out_d);

    return out_d;
}


/* -------------------------------------------------------------------------
 * DESCR   :	Parallel & perpendicular - use RNline<T> routines
 * ------------------------------------------------------------------------- */
template <class T>
inline
WINbool
R2Line_segTC<T>::IsParallelSeg(const R2Line_segTC<T> &in_l1, const R2Line_segTC<T> &in_l2)
{
	const T dDot = Dot(UnitSafe(in_l1.P2() - in_l1.P1()), UnitSafe(in_l2.P2() - in_l2.P1()));

	if ( RNApproxEqual(dDot, 1.0) )
		return TRUE;

	if ( RNApproxEqual(dDot, -1.0) )
		return TRUE;


    return FALSE;
}

template <class T>
inline
WINbool
R2Line_segTC<T>::IsPerpendicularSeg(const R2Line_segTC<T> &in_l1, const R2Line_segTC<T> &in_l2) 
{
	const T dDot = Dot(UnitSafe(in_l1.P2() - in_l1.P1()), UnitSafe(in_l2.P2() - in_l2.P1()));

	if ( RNApproxEqual(dDot, 0.0) )
		return TRUE;

    return FALSE;
}



/* -------------------------------------------------------------------------
 * DESCR   :	Compare two lines
 * ------------------------------------------------------------------------- */
template <class T>
inline
WINbool R2Line_segTC<T>::operator==(const R2Line_segTC<T> &in_l) const 
{
    if ( P1() == in_l.P1() && P2() == in_l.P2())
        return TRUE;
    
    if (P1() == in_l.P2() && P2() == in_l.P1())
        return TRUE;
    
    return FALSE;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Calculate the length of the line segment
 * ------------------------------------------------------------------------- */
template <class T>
inline
T R2Line_segTC<T>::Length() const
{
    // Get around stupid compiler
    return ::Length( R2VectorTC<T>( P2() - P1() ) );
}


template <class T>
inline
R2PointTC<T> R2Line_segTC<T>::operator()( T s) const 
{ 
    return m_pt1 + (m_pt2 - m_pt1) * s; 
}

template<class T>
T  R2Line_segTC<T>::Slope()  const 
{ 
    return (P1()[0] - P2()[0]) == 0 ? 0 :
        (P1()[1] - P2()[1]) / (P1()[0] - P2()[0]); 
}

template<class T>
WINbool R2Line_segTC<T>::Vertical()   const 
{ 
    return (RNIsZero(P1()[0] - P2()[0])) ?  TRUE :  FALSE; 
}


template<class T>
WINbool R2Line_segTC<T>::Horizontal() const 
{ 
    return (RNIsZero(P1()[1] - P2()[1])) ?  TRUE :  FALSE; 
}

/* -------------------------------------------------------------------------
 * DESCR   :	Set two lines equal
 * ------------------------------------------------------------------------- */
template<class T>
R2Line_segTC<T> & R2Line_segTC<T>::operator=(const R2Line_segTC<T> &in_l)
{
    P1() = in_l.P1();
    P2() = in_l.P2();
    
    return *this;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Construct a line segment perpendicular to the current line
 * 		segment with one point on the line segment, the other at
 * 		the given point
 * ------------------------------------------------------------------------- */
template<class T>
R2Line_segTC<T> 
R2Line_segTC<T>::Perpendicular(const R2PointTC<T> &in_p) const
{
    R2Line l1(P1(), P2());
    R2Line perp = l1.Perpendicular(in_p);

    return R2Line_segTC<T>(in_p, in_p + perp.Vec());
}

template<class T>
R2Line_segTC<T> 
R2Line_segTC<T>::Parallel(const R2PointTC<T> &in_p) const
{
    return R2Line_segTC<T>(in_p, in_p + (P2() - P1()));
}

/*
               p0y v1x - p1y v1x + (-p0x + p1x) v1y
Out[4]= {{s -> ------------------------------------, 
                       -(v0y v1x) + v0x v1y
 
           p0y v0x - p1y v0x + (-p0x + p1x) v0y
>     t -> ------------------------------------}}
                   -(v0y v1x) + v0x v1y

*/
template<class T>
WINbool R2Line_segTC<T>::Intersect( const R2Line_segTC<T> & in_l, 
                                    R2PointTC<T>          & out_pt,
                                    T                     & out_s, 
                                    T                     & out_t) const
{
	const R2VectorTC<T> vec1 = P2() - P1();
	const R2VectorTC<T> vec2 = in_l.P2() - in_l.P1();

	const T dDenom = vec1[0] * vec2[1] - vec1[1] * vec2[0];

	if ( RNIsZero( dDenom ) ) {
		T dDist = 0.0;
		out_t = 0.0;
		FindPtOnSeg( in_l.P1(), out_pt, out_s, dDist );
		if ( RNIsZero( dDist ) )
			return TRUE;

		out_t = 1.0;
		FindPtOnSeg( in_l.P2(), out_pt, out_s, dDist );
		if ( RNIsZero( dDist ) )
			return TRUE;
		return FALSE;
	}
	
	out_s = ( P1()[1] - in_l.P1()[1] ) * vec2[0] + ( in_l.P1()[0] - P1()[0] ) * vec2[1];
	out_s = out_s / dDenom;

	if ( RNIsZero( vec2[0] ) == FALSE ) {
		out_t =  ( P1()[0] + out_s * vec1[0] - in_l.P1()[0] ) / vec2[0];
	} else if ( RNIsZero( vec2[1] ) == FALSE ) {
		out_t =  ( P1()[1] + out_s * vec1[1] - in_l.P1()[1] ) / vec2[1];
	} else {
		out_t = 0;
	}
	out_pt = P1() + vec1 * out_s;
	const R2PointTC<T> ptCheck = in_l.P1() + vec2 * out_t;

	if ( out_s > RNEpsilon_d && out_s < 1.0 + RNEpsilon_d && out_t > RNEpsilon_d && out_t < 1.0 + RNEpsilon_d ) {
		return TRUE;
	}
	return FALSE;
}

template<class T>
T 
R2Line_segTC<T>::Projected_dist_on_seg(const R2PointTC<T> &in_pt) const
{
	const R2VectorTC<T> vec = P2() - P1();
	const T dLenVec = ::LengthSq( vec );
	if ( RNIsZero( dLenVec ) ) {
		return 0.0;
	}

	return Dot( in_pt - P1(), vec ) / (dLenVec);
}



template<class T>
WINbool
R2Line_segTC<T>::FindPtOnSeg( const R2PointTC<T> & in_p, 
                              R2PointTC<T>       & out_ptClosest, 
                              T                  & out_t_on_seg, 
                              T                  & out_d_to_seg) const
{
	const R2VectorTC<T> vec = P2() - P1();
	const T dLenVec = ::LengthSq( vec );
	if ( RNIsZero( dLenVec ) ) {
		out_ptClosest = P1();
		out_t_on_seg = 0;
		out_d_to_seg = ::Length( in_p - P1() );
		if ( RNIsZero( out_d_to_seg ) )
			return TRUE;
		return FALSE;
	}

	out_t_on_seg = Dot( in_p - P1(), vec ) / (dLenVec);

	if ( out_t_on_seg < -RNEpsilon_d ) {
		out_t_on_seg = 0.0;
		out_ptClosest = P1();
		out_d_to_seg = ::Length( in_p - P1() );
		return FALSE;
	}

	if ( out_t_on_seg > 1.0 + RNEpsilon_d ) {
		out_t_on_seg = 1.0;
		out_ptClosest = P2();
		out_d_to_seg = ::Length( in_p - P2() );
		return FALSE;
	}

	out_ptClosest = P1() + vec * out_t_on_seg;
	out_d_to_seg = ::Length( out_ptClosest - in_p );

	//const R2VectorTC<T> vecBack = out_ptClosest - in_p;
	//ASSERT( RNIsZero( Dot( vec, vecBack ) ) );

	if ( RNIsZero( out_d_to_seg ) )
		return TRUE;
    return FALSE;
}
                                         
template<class T>
R2PointTC<T> R2Line_segTC<T>::Project(const R2PointTC<T> &in_pt) const
{
	const R2VectorTC<T> vec = P2() - P1();
	const T dLenVec = ::LengthSq( vec );
	if ( RNIsZero( dLenVec ) ) {
		return P1();
	}

	const T dT = Dot( in_pt - P1(), vec ) / (dLenVec);

	if ( dT < -RNEpsilon_d ) {
		return P1();
	}

	if ( dT > 1.0 + RNEpsilon_d ) {
		return P2();
	}

	return P1() + vec * dT;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Print this
 * ------------------------------------------------------------------------- */
template<class T>
void R2Line_segTC<T>::Print() const 
{
    cout << "RNline_seg " << Dim() << " R2PointTC<T>s: " << P1() << P2() << "\n";
}

template<class T>
void R2Line_segTC<T>::Write(ofstream &out) const 
{
    out << P1() << " " << P2() << "\n";
}

template<class T>
WINbool R2Line_segTC<T>::Read(ifstream &in) 
{
    WINbool bRes1 = P1().Read(in);
    WINbool bRes2 = P2().Read(in);
    return (bRes1 && bRes2) ? TRUE : FALSE;
}
