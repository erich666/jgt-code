#ifndef _RN_DEFS_LINE_SEG_H
#define _RN_DEFS_LINE_SEG_H

#include "R2_Line.H"

/** A 2D line segment.
    Defined by two points.
<br> Files: 
<dd><li> include/utils/Rn_Line_seg.H <br>
<dd><li> include/utils/R2_Line_seg_i.H <br>
<dd><li> utils/geometry/R2Line_seg.cpp <br>
*/
template <class T>
class R2Line_segTC  {
protected:
    R2PointTC<T> m_pt1, m_pt2;
    
public:
    /**@name Access to data */
    //@{
    /// Constant function returns 2
    inline int Dim()          const { return 2; }
    ///
    inline const R2PointTC<T>   &P1() const { return m_pt1; }
    ///
    inline const R2PointTC<T>   &P2() const { return m_pt2; }
    ///
    inline       R2PointTC<T>   &P1()       { return m_pt1; }
    ///
    inline       R2PointTC<T>   &P2()       { return m_pt2; }
    //@}

    /**@name Queries: Comparison and type */
    //@{
    ///
    WINbool        Vertical() const;
    ///
    WINbool        Horizontal() const;
    /// Are the two line segments parallel?
    static WINbool IsParallelSeg(const R2Line_segTC<T> &, const R2Line_segTC<T> &) ;
  //    friend WINbool IsParallelSeg(const R2Line_segTC<T> &, const R2Line_segTC<T> &) ;
    /// Are the two line segments perpendicular?
    static WINbool IsPerpendicularSeg(const R2Line_segTC<T> &, const R2Line_segTC<T> &);
  //    friend WINbool PerpendicularSeg(const R2Line_segTC<T> &, const R2Line_segTC<T> &);
    ///
    WINbool        operator==(const R2Line_segTC<T> &) const;
    //@}

    /**@name Calculations on the segment */
    //@{
    /** Point on the segment.
        @return $s * p_1 + (1-s) * p_2$
    */
    R2PointTC<T> operator()( T s) const ;
    ///
    T            Length() const ;
    ///
    T            Slope()  const;
    //@}
    
    /**@name Project point onto line segment */
    //@{
    /** Project the point onto the line segement, clipping at the end points.
       @param in_pt Point to project
       @param out_ptClosest The closest point
       @param out_t_on_line Parameter of closest point, i.e., operator(t) = closest point. Clips to [0,1].
       @param out_d_to_line Distance point is from line
       @return TRUE if point is on the line.
    */
    WINbool     FindPtOnSeg( const R2PointTC<T> & in_pt, 
                             R2PointTC<T>       & out_ptClosest, 
                             T     & out_t_on_seg, 
                             T     & out_d_to_seg) const;
    /// Is the point within epsilon of the line segment? 
    WINbool      IsPtOnSeg   ( const R2PointTC<T> &in_pt ) const ;
    /// Distance from given point to closest point on line segment
    T            Dist_to_seg ( const R2PointTC<T> &in_pt ) const ;
    /// Closest point on segment
    R2PointTC<T> Project     ( const R2PointTC<T> &in_pt ) const;
    /// t between 0 and 1
    T            Projected_dist_on_seg( const R2PointTC<T> &in_pt ) const;
    //@}

    /**@name Intersection routines */
    //@{
    /** Intersect this line segment with another line segment.
        @param in_lineSeg The other line segment.
        @param out_pt The intersection point, if exists.
        @param out_s  The parameter value of the intersection point for this line segment.
        @param out_t  The parameter value of the intersection point for in_lineSeg
        @return TRUE if the line segments intersect.
        */
    WINbool Intersect( const R2Line_segTC<T> & in_lineSeg, 
                       R2PointTC<T>          & out_pt,
                       T                     & out_s, 
                       T                     & out_t) const;
    //@}
    
    /**@name Construct line segments */
    //@{
    /** Make a line segment perpendicular to this one.
        One end is on this line segment (or line) and the other is at the input point */
    R2Line_segTC<T>   Perpendicular ( const R2PointTC<T> &in_pt ) const ;
    /** Make a line segment parallel to this one.
        The new p1 is the input point, the new p2 is at the new p1 + the old (p2-p1) */
    R2Line_segTC<T>   Parallel      ( const R2PointTC<T> &in_pt ) const ;
    /// 
    R2Line_segTC<T>  &operator=     ( const R2Line_segTC<T> &in_line );
    /// Default constructor. Unit line segment.
    inline R2Line_segTC() : m_pt1(0,0), m_pt2(0,1) {}
    /// Construct from two points.
    inline R2Line_segTC( const R2PointTC<T> &p1, const R2PointTC<T> &p2 ) : m_pt1(p1), m_pt2(p2) {}
    /// Construct from another line.
    inline R2Line_segTC( const R2Line_segTC<T> &in_l ) { (*this) = in_l; }
    //@}

    ///
    inline ~R2Line_segTC() {}

    /**@name Read, write, print */
    //@{
    /// Debugging print.
    void    Print() const ;
    /// Writes 2 points to file.
    void    Write ( ofstream &out ) const ;
    /// Reads 2 points from file.
    WINbool Read  ( ifstream &in );
    //@}

    /// Test routine
    static WINbool Test();
};



#include "R2_Line_seg_i.H"

typedef R2Line_segTC<double> R2Line_seg;

#endif

