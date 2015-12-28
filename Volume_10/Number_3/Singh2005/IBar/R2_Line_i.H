/* -------------------------------------------------------------------------
 * DESCR   :	Is the point on the line?
 * ------------------------------------------------------------------------- */
inline WINbool R2Line::IsPtOnLine(const R2Pt &in_pt) const
{
    if (Vertical() == TRUE) {
        if (RNIsZero(in_pt[0] - Intercept())) return TRUE;
        else                                  return FALSE;
    }
    double y = Slope() * in_pt[0] + Intercept();
    
    if ( RNIsZero( y - in_pt[1] ) ) return TRUE;
    return FALSE;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Variations on FindPtOnLine
 * ------------------------------------------------------------------------- */
inline double R2Line::Dist_to_line(const R2Pt &in_pt) const
{
   R2Pt out_pt;
   double out_t, out_d;
   FindPtOnLine(in_pt, out_pt, out_t, out_d);

   return out_d;
}

inline R2Pt R2Line::Project(const R2Pt &in_pt) const
{
    R2Pt out_pt;
    double out_t, out_d;
    FindPtOnLine(in_pt, out_pt, out_t, out_d);

    return out_pt;
}

inline
double R2Line::Projected_dist_on_line(const R2Pt &in_pt) const
{
    R2Pt out_pt;
    double out_t, out_d;
    FindPtOnLine(in_pt, out_pt, out_t, out_d);

    return out_t;
}


/* -------------------------------------------------------------------------
 * DESCR   :	Determine if the two lines are parallel
 * ------------------------------------------------------------------------- */
inline
WINbool IsParallel(const R2Line &in_l1, const R2Line &in_l2) 
{
    double res = 1.0f - (double) fabs( Dot(in_l1.Vec(), in_l2.Vec()) );
    if ( RNIsZero( res ) )
        return TRUE;

    return FALSE;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Determine if the two lines are perpendicular
 * ------------------------------------------------------------------------- */
inline
WINbool IsPerpendicular(const R2Line &in_l1, const R2Line &in_l2)
{
    if ( RNIsZero( Dot( in_l1.Vec(), in_l2.Vec() ) ) )
        return TRUE;
    
    return FALSE;
}


/* -------------------------------------------------------------------------
 * DESCR   :	Comparison
 * ------------------------------------------------------------------------- */
inline WINbool R2Line::operator==(const R2Line &in_l) const 
{
    if (ApproxEqual(Vec(), in_l.Vec()) == FALSE) return FALSE;
    
    if (ApproxEqual(Pt(), in_l.Pt()) == TRUE) return TRUE;
    
    return FALSE;
}


inline double R2Line::Slope() const 
{ 
    const R2Pt pt = m_pt + m_vec;
    if ( RNIsZero( m_pt[0] - pt[0] ) )
        return 1e30;

    return (m_pt[1] - pt[1]) / (m_pt[0] - pt[0]);
}

/* -------------------------------------------------------------------------
 * DESCR   :	Return a perpendicular line
 * ------------------------------------------------------------------------- */
double  R2Line::Intercept()  const
{
    return m_pt[1] - Slope() * m_pt[0];
}



/* -------------------------------------------------------------------------
 * DESCR   :	Copy a line
 * ------------------------------------------------------------------------- */
inline R2Line::R2Line() : m_pt(0,0), m_vec(0,1)
{
}

inline void R2Line::SetVec(const R2Vec &in_vec) 
{ 
    ASSERT( ! RNIsZero( Length( m_vec ) ) );
    m_vec = UnitSafe(in_vec); 
}

inline R2Line::R2Line(const R2Line &in_l) 
{
    (*this) = in_l;
}

inline
R2Line::R2Line(const R2Pt &in_pt, const R2Vec &in_vec)
: m_pt( in_pt ), m_vec( UnitSafe(in_vec) )
{
    ASSERT( ! RNIsZero( Length( m_vec ) ) );
}



inline 
R2Line::R2Line(const R2Pt &in_pt1, const R2Pt  &in_pt2) 
    : m_pt( in_pt1 ), m_vec( UnitSafe( in_pt2 - in_pt1 ) )
{
    ASSERT( ! RNIsZero( Length( m_vec ) ) );
}

