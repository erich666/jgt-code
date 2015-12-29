/* clip.c - */

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

#include "gts.h"

#include "clip.h"

#define printf
#define g_assert(x) 

#define linear(alpha,a,b) (alpha*((a)-(b))+(b))

GtsPoint *intersect_y1(GtsPoint *p1, GtsPoint *p2)
{
  double alpha = (1.0 - p2->y) / (p1->y - p2->y);
  return (gts_point_new(gts_point_class(), linear(alpha, p1->x, p2->x), 1.0, 0.0));
}

GtsPoint *intersect_y0(GtsPoint *p1, GtsPoint *p2)
{
  double alpha = (-p2->y) / (p1->y - p2->y);
  return (gts_point_new(gts_point_class(), linear(alpha, p1->x, p2->x), 0.0, 0.0));
}

GtsPoint *intersect_x1(GtsPoint *p1, GtsPoint *p2)
{
  double alpha = (1.0 - p2->x) / (p1->x - p2->x);
  return (gts_point_new(gts_point_class(), 1.0, linear(alpha, p1->y, p2->y), 0.0));
}

GtsPoint *intersect_x0(GtsPoint *p1, GtsPoint *p2)
{
  double alpha = (-p2->x) / (p1->x - p2->x);
  return (gts_point_new(gts_point_class(), 0.0, linear(alpha, p1->y, p2->y), 0.0));
}

GSList *add_to_tri_list(GSList *tris, GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  GtsVertex *v1, *v2, *v3;
  GtsEdge *e1, *e2, *e3;
  GtsTriangle *t;

  v1 = gts_vertex_new(gts_vertex_class(), p1->x, p1->y, 0.0);
  v2 = gts_vertex_new(gts_vertex_class(), p2->x, p2->y, 0.0);
  v3 = gts_vertex_new(gts_vertex_class(), p3->x, p3->y, 0.0);

  e1 = gts_edge_new(gts_edge_class(), v1, v2);
  e2 = gts_edge_new(gts_edge_class(), v2, v3);
  e3 = gts_edge_new(gts_edge_class(), v3, v1);
  
  t = gts_triangle_new(gts_triangle_class(), e1, e2, e3);

  return (g_slist_prepend(tris, t));
}

GSList *clip_y1(GSList *tris,
                     GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  g_assert(p1->y <= 1.0);

  // clip at y = 1.0
  int num_clipped = 
    ((p2->y > 1.0) ? 1 : 0) + 
    ((p3->y > 1.0) ? 1 : 0);

  if (num_clipped == 0) return (add_to_tri_list(tris, p1, p2, p3));

  if (num_clipped == 2) {
    GtsPoint *intersect12, *intersect13;

    intersect12 = intersect_y1(p1, p2);
    intersect13 = intersect_y1(p1, p3);
    tris = add_to_tri_list(tris, p1, intersect12, intersect13);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect13));
    return (tris);
  }

  // one vertex clipped
  // keep p1 as first vertex
  if (p2->y > 1.0) {
    GtsPoint *intersect12, *intersect23;
    intersect12 = intersect_y1(p1, p2);
    intersect23 = intersect_y1(p2, p3);
    tris = add_to_tri_list(tris, p1, intersect12, intersect23);
    tris = add_to_tri_list(tris, p1, intersect23, p3);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  } else {
    GtsPoint *intersect31, *intersect23;
    intersect31 = intersect_y1(p3, p1);
    intersect23 = intersect_y1(p2, p3);
    tris = add_to_tri_list(tris, p1, p2, intersect23);
    tris = add_to_tri_list(tris, p1, intersect23, intersect31);
    gts_object_destroy(GTS_OBJECT(intersect31));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  }
}

GSList *clip_y0(GSList *tris,
                     GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  g_assert(p1->y >= 0.0);


  // clip at y = 0.0
  int num_clipped = 
    ((p2->y < 0.0) ? 1 : 0) + 
    ((p3->y < 0.0) ? 1 : 0);

  if (num_clipped == 0) return (clip_y1(tris, p1, p2, p3));

  if (num_clipped == 2) {
    GtsPoint *intersect12, *intersect13;

    intersect12 = intersect_y0(p1, p2);
    intersect13 = intersect_y0(p1, p3);
    tris = clip_y1(tris, p1, intersect12, intersect13);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect13));
    return (tris);
  }

  // one vertex clipped
  // keep p1 as first vertex
  if (p2->y < 0.0) {
    GtsPoint *intersect12, *intersect23;
    intersect12 = intersect_y0(p1, p2);
    intersect23 = intersect_y0(p2, p3);
    tris = clip_y1(tris, p1, intersect12, intersect23);
    tris = clip_y1(tris, p1, intersect23, p3);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  } else {
    GtsPoint *intersect31, *intersect23;
    intersect31 = intersect_y0(p3, p1);
    intersect23 = intersect_y0(p2, p3);
    tris = clip_y1(tris, p1, p2, intersect23);
    tris = clip_y1(tris, p1, intersect23, intersect31);
    gts_object_destroy(GTS_OBJECT(intersect31));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  }
}

GSList *clip_x1(GSList *tris,
                     GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  g_assert(p1->x <= 1.0);

  // clip at x = 1.0
  int num_clipped = 
    ((p2->x > 1.0) ? 1 : 0) + 
    ((p3->x > 1.0) ? 1 : 0);

  if (num_clipped == 0) return (clip_y0(tris, p1, p2, p3));

  if (num_clipped == 2) {
    GtsPoint *intersect12, *intersect13;

    intersect12 = intersect_x1(p1, p2);
    intersect13 = intersect_x1(p1, p3);
    tris = clip_y0(tris, p1, intersect12, intersect13);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect13));
    return (tris);
  }

  // one vertex clipped
  // keep p1 as first vertex
  if (p2->x > 1.0) {
    GtsPoint *intersect12, *intersect23;
    intersect12 = intersect_x1(p1, p2);
    intersect23 = intersect_x1(p2, p3);
    tris = clip_y0(tris, p1, intersect12, intersect23);
    tris = clip_y0(tris, p1, intersect23, p3);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  } else {
    GtsPoint *intersect31, *intersect23;
    intersect31 = intersect_x1(p3, p1);
    intersect23 = intersect_x1(p2, p3);
    tris = clip_y0(tris, p1, p2, intersect23);
    tris = clip_y0(tris, p1, intersect23, intersect31);
    gts_object_destroy(GTS_OBJECT(intersect31));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  }
}

GSList *clip_x0(GSList *tris,
                     GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  g_assert(p1->x >= 0.0);


  // clip at x = 0.0
  int num_clipped = 
    ((p2->x < 0.0) ? 1 : 0) + 
    ((p3->x < 0.0) ? 1 : 0);

  if (num_clipped == 0) return (clip_x1(tris, p1, p2, p3));

  if (num_clipped == 2) {
    GtsPoint *intersect12, *intersect13;

    intersect12 = intersect_x0(p1, p2);
    intersect13 = intersect_x0(p1, p3);
    tris = clip_x1(tris, p1, intersect12, intersect13);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect13));
    return (tris);
  }

  // one vertex clipped
  // keep p1 as first vertex
  if (p2->x < 0.0) {
    GtsPoint *intersect12, *intersect23;
    intersect12 = intersect_x0(p1, p2);
    intersect23 = intersect_x0(p2, p3);
    tris = clip_x1(tris, p1, intersect12, intersect23);
    tris = clip_x1(tris, p1, intersect23, p3);
    gts_object_destroy(GTS_OBJECT(intersect12));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  } else {
    GtsPoint *intersect31, *intersect23;
    intersect31 = intersect_x0(p3, p1);
    intersect23 = intersect_x0(p2, p3);
    tris = clip_x1(tris, p1, p2, intersect23);
    tris = clip_x1(tris, p1, intersect23, intersect31);
    gts_object_destroy(GTS_OBJECT(intersect31));
    gts_object_destroy(GTS_OBJECT(intersect23));
    return (tris);
  }
}

GSList *clip(GSList *tris,
             GtsPoint *p1, GtsPoint *p2, GtsPoint *p3)
{
  return (clip_x0(tris, p1, p2, p3));
}
