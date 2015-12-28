#include "StdAfx.H"
#include "R2_Line_seg.H"


WINbool R2Line_segTC<double>::Test()
{
    R2Pt pt1d(0,0.1), pt2d(2,0);
    R2Vec vecd(2, 0);
    R2Line_seg line1d, line3d;
    R2Line_seg line2d(pt1d, pt1d + vecd);
    
    line1d = line2d;
    VERIFY(line2d.Dim() == 2);
    VERIFY(line1d == line2d);
    VERIFY(::Length(line1d.P2() - line1d.P1()) == ::Length(vecd));
    VERIFY(line1d.Length() == ::Length(vecd));
    
    pt2d = line1d(0.2);
    VERIFY(pt2d[0] == 0.4 && pt2d[1] == 0.1 && pt2d[2] == 0);
    VERIFY(IsParallelSeg(line1d, line2d));
    
    double out_t_on_line = -1, out_d_to_line = -1;
    vecd = R2Vec(0, 0.1);
    pt1d = pt2d + vecd;
    VERIFY(line1d.FindPtOnSeg(pt1d, pt2d, out_t_on_line, out_d_to_line) == FALSE);
    VERIFY(pt2d == line1d(0.2));
    ASSERT(out_t_on_line == 0.2);
    VERIFY(out_d_to_line == ::Length(vecd) );
    
    pt2d = line1d(1.2);
    pt1d = pt2d;
    VERIFY(line1d.FindPtOnSeg(pt1d, pt2d, out_t_on_line, out_d_to_line) == FALSE);
    VERIFY(pt2d == line1d.P2());
    ASSERT(out_t_on_line == 1.0);
    VERIFY( RNApproxEqual( out_d_to_line, ::Length( line1d(1.2) - line1d.P2())));
    
    line1d.Print();
    ofstream out("RNtest.txt", ios::out);
    line1d.Write(out);
    out.close();
    ifstream in("RNtest.txt", ios::in);
    line3d.Read(in);
    VERIFY(line1d == line3d);
    
    R2Pt pt1(0,0.1), pt2(1,0);
    R2Vec vec(0, 0.2);
    R2Line_seg line1, line2(pt1, pt2), line3(pt1, pt1 + vec);
    
    line1 = line2;
    VERIFY(line1 == line2);
    line1 = line2.Perpendicular(pt2);
    VERIFY(IsPerpendicularSeg(line1, line2));
    
    line1 = line2.Parallel(pt2);
    VERIFY(IsParallelSeg(line1, line2));
    
    VERIFY(IsPerpendicularSeg(line2, line3) == FALSE);
    VERIFY(IsParallelSeg(line2, line3) == FALSE);
    
    VERIFY(!(line1 == line3));
    
    VERIFY(line2.Dim() == 2);
    VERIFY(RNApproxEqual(line3.Length(), ::Length(vec)));
    
   
    line1d = R2Line_seg(R2Pt(0.1, 0), 
                        R2Pt(0.1, 0) + R2Vec(0, 1.0));
    
    pt2 = line1d(0.2);
    VERIFY(pt2[0] == 0.1 && pt2[1] == 0.2);
    out_t_on_line = -1;
    out_d_to_line = -1;
    vec = R2Vec(0.1, 0.0);
    pt1 = pt2 + vec;
    VERIFY(line1d.FindPtOnSeg(pt1, pt2, out_t_on_line, out_d_to_line) == FALSE);
    VERIFY(line1d.IsPtOnSeg(pt2) == TRUE);
    VERIFY(pt2 == line1d(0.2));
    ASSERT(out_t_on_line == 0.2);
    VERIFY(out_d_to_line == ( ::Length(vec) ));
    
    VERIFY(line1d.Slope() == 0.0);
    VERIFY(line1d.Vertical());
    VERIFY(line1d.Horizontal() == FALSE);
    
    R2Line_seg l(R2Pt(-0.3, 0.0), R2Pt(0.1, 0.3));
    R2Line_seg lv(R2Pt(0.1, 0.0), R2Pt(0.1, 0.6));
    R2Line_seg lh(R2Pt(-1, 0.2), R2Pt(1, 0.2));
    
    double s = -1, t= -1;
    VERIFY(lv.Vertical());
    VERIFY(!lv.Horizontal());
    VERIFY(!l.Vertical());
    VERIFY(!lv.Horizontal());
    VERIFY(lh.Horizontal());
    VERIFY(!lh.Vertical());
    VERIFY(lh.Intersect(lv, pt2, s, t));
    VERIFY( ApproxEqual( pt2, R2Pt(0.1, 0.2) ) );
    VERIFY(RNApproxEqual(t, 1.0/3.0));
    VERIFY(RNApproxEqual(s, 1.1 / 2.0));
    
    VERIFY(lv.Intersect(lh, pt2, s, t));
    VERIFY(pt2[0] == 0.1 && pt2[1] == 0.2);
    VERIFY(RNApproxEqual(s, 1.0/3.0));
    VERIFY(RNApproxEqual(t, 1.1 / 2.0));
    
    VERIFY(lv.Intersect(l, pt2, s, t));
    VERIFY(RNApproxEqual(pt2[0], 0.1) && RNApproxEqual(pt2[1], 0.3));
    VERIFY(RNApproxEqual(s, 0.5));
    VERIFY(RNApproxEqual(t, 1.0));
    
    VERIFY(l.Intersect(lv, pt2, s, t));
    VERIFY(RNApproxEqual(pt2[0], 0.1) && RNApproxEqual(pt2[1], 0.3));
    VERIFY(RNApproxEqual(t, 0.5));
    VERIFY(RNApproxEqual(s, 1.0));
    
    VERIFY(lh.Intersect(l, pt2, s, t));
    VERIFY(RNApproxEqual(pt2[1], 0.2));
    
    VERIFY(l.Intersect(lh, pt2, s, t));
    VERIFY(RNApproxEqual(pt2[1], 0.2));
    
    
    R2Line_seg lp = l.Parallel(R2Pt(-0.1, 0.1));
    VERIFY(l.Intersect(lp, pt2, s, t) == FALSE);
    
    lp = R2Line_seg(l(0.2) - R2Vec(-2, 3), 
			    l(0.2) + R2Vec(-2, 3));
    VERIFY(l.Intersect(lp, pt2, s, t));
    VERIFY(ApproxEqual(pt2, l(0.2)));
    
    return TRUE;
}

