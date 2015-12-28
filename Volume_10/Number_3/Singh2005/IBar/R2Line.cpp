#include "StdAfx.H"
#include "R2_Line.H"

WINbool R2Line::FindPtOnLine( const R2Pt  & in_pt,
                              R2Pt        & out_ptClosest,
                              double      & out_t_on_line,
                              double      & out_d_to_line) const
{
    // Move start of pt-vector to origin, then project onto the vector

    out_t_on_line = Dot( in_pt - m_pt, m_vec );
   
    out_ptClosest = m_pt + m_vec * out_t_on_line;
    
    out_d_to_line = Length(out_ptClosest - in_pt);

    return RNIsZero(out_d_to_line);
}

/* -------------------------------------------------------------------------
 * DESCR   :	Return a perpendicular line
 * ------------------------------------------------------------------------- */
R2Line
R2Line::Perpendicular(const R2Pt &in_pt) const 
{
    R2Vec vec(Vec()[1], -Vec()[0]);
    return R2Line(in_pt, vec);
}

/* -------------------------------------------------------------------------
 * DESCR   :	Return a parallel line
 * ------------------------------------------------------------------------- */
R2Line
R2Line::Parallel(const R2Pt &in_pt) const 
{
    return R2Line(in_pt, Vec());
}

/* -------------------------------------------------------------------------
 * DESCR   :	Return the corresponding x point, if it exists
 * ------------------------------------------------------------------------- */
double
R2Line::X(double in_dY) const 
{
    if (Horizontal() == TRUE)
        return 0;
    
    if (Vertical() == TRUE)
        return Intercept();
    
    return (in_dY - Intercept()) / Slope();
}

/* -------------------------------------------------------------------------
 * DESCR   :	Return the corresponding y point, if it exists
 * ------------------------------------------------------------------------- */
double R2Line::Y(double in_dX) const 
{
    if (Horizontal() == TRUE)
        return Intercept();
    
    if (Vertical() == TRUE)
        return 0;
    
    return Slope() * in_dX + Intercept();
}



/* -------------------------------------------------------------------------
 * DESCR   :	Intersect the two lines
 * ------------------------------------------------------------------------- */
WINbool
R2Line::Intersect(const R2Line &in_l, R2Pt &out_pt) const 
{
    out_pt = R2Pt(1e30f, 1e30f);
    if (IsParallel(*this, in_l) == TRUE)
        return FALSE;
    
    if (Vertical() == TRUE) { 
        out_pt[0] = Intercept();
        out_pt[1] = in_l.Y(Intercept());
        return TRUE;
    }
    
    if (in_l.Vertical() == TRUE) {
        out_pt[0] = in_l.Intercept();
        out_pt[1] = Y(in_l.Intercept());
        return TRUE;
    }
    
    out_pt[0] = (in_l.Intercept() - Intercept()) / (Slope() - in_l.Slope());
    out_pt[1] = Slope() * out_pt[0] + Intercept();
    
    return TRUE;
}

/* -------------------------------------------------------------------------
 * DESCR   :	Set two lines equal
 * ------------------------------------------------------------------------- */
R2Line &
R2Line::operator=(const R2Line &in_l)
{
    m_pt = in_l.m_pt;
    m_vec = in_l.m_vec;
    
    return *this;
}



/* -------------------------------------------------------------------------
 * DESCR   :	Make a vertical line
 * ------------------------------------------------------------------------- */
R2Line::R2Line(double x_intercept) 
{
    m_pt[0] = x_intercept;
    m_pt[1] = 0.0;
    m_vec[0] = 0.0;
    m_vec[1] = 1.0;
    }

/* -------------------------------------------------------------------------
 * DESCR   :	Initialize with slope and intercept
 * ------------------------------------------------------------------------- */
R2Line::R2Line( double in_dSlope,
                double in_dIntercept) 
{
    m_pt[0] = 0.0;
    m_pt[1] = in_dIntercept;
    
    R2Pt p2;
    p2[0] = 1.0;
    p2[1] = in_dSlope + in_dIntercept;
    m_vec = (p2 - Pt());
    m_vec = UnitSafe(Vec());

    ASSERT( !RNIsZero( Length( m_vec ) ) );
}

/* -------------------------------------------------------------------------
 * DESCR   :	Initialize with Ax + By + C
 * ------------------------------------------------------------------------- */
R2Line::R2Line( double in_dA,
                double in_dB,
                double in_dC) 
{
    // B y = - Ax - C
    m_pt = R2Pt(0,0);
    m_vec = R2Vec(1,0);

    if ( RNIsZero( in_dB ) && RNIsZero( in_dA ) ) {
        ASSERT(FALSE);
        return;
    } else if ( RNIsZero( in_dB ) ) {
        // Ax = - C
        m_pt[0] = -in_dC / in_dA;
        m_vec = R2Vec(1,0);

    } else if ( RNIsZero( in_dA ) ) {
        // By = - C
        m_pt[1] = -in_dC / in_dB;
        m_vec = R2Vec(0,1);
    } else {	// Fixed
		m_pt[1]  = -in_dC / in_dB;
		m_vec[1] = -in_dA / in_dB;
    }
    m_vec = UnitSafe(m_vec);

    ASSERT( RNIsZero( in_dA * ( m_pt[0] + m_vec[0] ) + in_dB * ( m_pt[1] + m_vec[1] ) + in_dC ) );

    ASSERT( !RNIsZero( Length( m_vec ) ) );
}

/* -------------------------------------------------------------------------
 * DESCR   :	Print this 
 * ------------------------------------------------------------------------- */
void R2Line::Print() const
{
    cout << "RNline" << Dim() << " point ";
    Pt().Print();
    cout << " vec ";
    Vec().Print();
    cout << "\n";
}

void R2Line::Write(ofstream &out) const
{
    out << Pt() << " " << Vec() << "\n";
}

WINbool R2Line::Read(ifstream &in)
{
    WINbool bRes1 = m_pt.Read(in);
    WINbool bRes2 = m_vec.Read(in);
    return (bRes1 && bRes2) ? TRUE : FALSE;
}

WINbool R2Line::Test()
{
   R2Pt pt1d(0,0.1), pt2d(2,0);
   R2Vec vecd(2, 0);
   R2Line line1d, line3d;
   R2Line line2d(pt1d, vecd);

   line1d = line2d;
   VERIFY(line2d.Dim() == 2);
   VERIFY(line1d == line2d);
   VERIFY(Length(line1d.Vec()) == 1.0);

   pt2d = line1d(0.2);
   VERIFY(pt2d[0] == 0.2 && pt2d[1] == 0.1);
   VERIFY(IsParallel(line1d, line2d));

   double out_t_on_line = -1, out_d_to_line = -1;
   vecd = R2Vec(0, 0.1);
   pt1d = pt2d + vecd;
   VERIFY(line1d.FindPtOnLine(pt1d, pt2d, out_t_on_line, out_d_to_line) == FALSE);
   VERIFY(pt2d == line1d(0.2));
   ASSERT(out_t_on_line == 0.2);
   VERIFY(out_d_to_line == Length(vecd));

   line1d.Print();
   ofstream out("RNtest.txt", ios::out);
   line1d.Write(out);
   out.close();
   ifstream in("RNtest.txt", ios::in);
   line3d.Read(in);
   VERIFY(line1d == line3d);

   R2Pt pt1(0,0.1), pt2(1,0);
   R2Vec vec(0, 0.2);
   R2Line line1, line2(pt1, pt2), line3(pt1, vec);

   line1 = line2;
   VERIFY(line1 == line2);
   line1 = line2.Perpendicular(pt2);
   VERIFY(IsPerpendicular(line1, line2));

   line1 = line2.Parallel(pt2);
   VERIFY(IsParallel(line1, line2));

   VERIFY(IsPerpendicular(line2, line3) == FALSE);
   VERIFY(IsParallel(line2, line3) == FALSE);

   VERIFY(!(line1 == line3));

   VERIFY(line2.Dim() == 2);
   VERIFY(Length(line1.Vec()) == 1.0);


   line1d = R2Line(R2Pt(0.1, 0), R2Vec(0, 1.0));
   pt2 = line1d(0.2);
   VERIFY(pt2[0] == 0.1 && pt2[1] == 0.2);
   out_t_on_line = -1;
   out_d_to_line = -1;
   vec = R2Vec(0.1, 0.0);
   pt1 = pt2 + vec;
   VERIFY(line1d.FindPtOnLine(pt1, pt2, out_t_on_line, out_d_to_line) == FALSE);
   VERIFY(line1d.IsPtOnLine(pt2) == TRUE);
   VERIFY(pt2 == line1d(0.2));
   ASSERT(out_t_on_line == 0.2);
   VERIFY(out_d_to_line == Length(vec));

   VERIFY(line1d.Slope() == 0.0);
   VERIFY(line1d.Intercept() == 0.1);
   VERIFY(line1d.Vertical());
   VERIFY(line1d.X(0.1) == 0.1);
   VERIFY(line1d.Y(0.1) == 0.0);
   VERIFY(line1d.Horizontal() == FALSE);

   R2Line l(3.0, 0.2), lv(0.1), lh(R2Pt(0.0, 0.2), 
                                              R2Vec(1.0, 0.0));
   VERIFY(lv.Vertical());
   VERIFY(!lv.Horizontal());
   VERIFY(!l.Vertical());
   VERIFY(!lv.Horizontal());
   VERIFY(lh.Horizontal());
   VERIFY(!lh.Vertical());
   VERIFY(lh.Intersect(lv, pt2));
   VERIFY(pt2[0] == 0.1 && pt2[1] == 0.2);

   VERIFY(lv.Intersect(lh, pt2));
   VERIFY(pt2[0] == 0.1 && pt2[1] == 0.2);

   VERIFY(lv.Intersect(l, pt2));
   VERIFY(pt2[0] == 0.1 && pt2[1] == 0.5);

   VERIFY(l.Intersect(lv, pt2));
   VERIFY(pt2[0] == 0.1 && pt2[1] == 0.5);

   VERIFY(lh.Intersect(l, pt2));
   VERIFY(pt2[0] == 0.0 && pt2[1] == 0.2);

   VERIFY(l.Intersect(lh, pt2));
   VERIFY(pt2[0] == 0.0 && pt2[1] == 0.2);


   R2Line lp = l.Parallel(R2Pt(-0.1, 0.1));
   VERIFY(l.Intersect(lp, pt2) == FALSE);

   lp = R2Line(l(0.2), R2Vec(-2, 3));
   VERIFY(l.Intersect(lp, pt2));
   VERIFY( ApproxEqual(pt2, l(0.2)) );

   return TRUE;
}

