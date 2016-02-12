/* advicer_algorithms.h*/

#include "triangle.h"
#include <math.h>

int intersect_triangle_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle1_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle2_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle3_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_small_baryCCW(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_mahovsky_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p);




int chirkov_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov3_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov2_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int plucker_mahovsky_other_bary(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int plucker_other_bary(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int orourke_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int orourke_other_baryCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int halfplane_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int area2D_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int arenberg_other_bary_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int arenberg_other_bary_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int halfplane_other_bary_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane_other_bary_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane2_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p);


int intersect_triangle_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle1_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle2_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle3_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_small_tCCW(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_mahovsky_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p); 

int chirkov_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov3_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov2_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int plucker_mahovsky_other_t(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int plucker_other_t(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int orourke_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int orourke_other_tCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int halfplane_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int area2D_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int arenberg_other_t_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int arenberg_other_t_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int halfplane_other_t_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane_other_t_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane2_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p);




int intersect_triangle_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle1_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle2_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int intersect_triangle3_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int orourke_smallCCW(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_mahovsky_small(Ray_big *r, Triangle_small *t, Intersection_big *p);
int plucker_small(Ray_big *r, Triangle_small *t, Intersection_big *p);

int chirkov_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov3_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int chirkov2_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int plucker_mahovsky_other(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int plucker_other(Ray_big *r, Plucker_coords *t, Intersection_big *p);
int orourke_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int orourke_otherCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int halfplane_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int area2D_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
int arenberg_other_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int arenberg_other_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p);
int halfplane_other_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane_other_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p);
int halfplane2_other(Ray_big *r, Triangle_plane *t, Intersection_big *p);
