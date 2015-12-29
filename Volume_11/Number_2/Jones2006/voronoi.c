/* voronoi.c - */

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

#include <math.h>
#include <stdlib.h>
#include "gts.h"

#include "voronoi.h"
#include "clip.h"
#include "triangle_area.h"
#define printf
#define g_assert(x) 

GSList *create_voronoi_region(GtsVertex *v)
{
  GSList *seg;
  GSList *out_tris = NULL;

  seg = v->segments;
  while (seg) {
    GtsTriangle *tri1 = GTS_TRIANGLE(GTS_EDGE(seg->data)->triangles->data);
    GtsTriangle *tri2 = GTS_TRIANGLE(GTS_EDGE(seg->data)->triangles->next->data);
    GtsPoint *p1, *p2;
    g_assert(tri1);
    g_assert(tri2);
    p1 = gts_triangle_circumcircle_center(tri1, gts_point_class());
    p2 = gts_triangle_circumcircle_center(tri2, gts_point_class());
    out_tris = clip(out_tris, GTS_POINT(v), p1, p2);
    gts_object_destroy(GTS_OBJECT(p1));
    gts_object_destroy(GTS_OBJECT(p2));
    seg = seg->next;
  }

  return (out_tris);
}

static int same_point(GtsVertex *v1, GtsVertex *v2)
{
  return ((GTS_POINT(v1)->x == GTS_POINT(v2)->x) &&
          (GTS_POINT(v1)->y == GTS_POINT(v2)->y));
}

int one_vertex_is(GtsTriangle *t, GtsVertex *v)
{
  GtsVertex *v1, *v2, *v3;

  gts_triangle_vertices(t, &v1, &v2, &v3);

  return (same_point(v, v1) ||
          same_point(v, v2) ||
          same_point(v, v3));
}

double compute_voronoi_area(GtsVertex *v, double exclude_radius)
{
  double area = 0.0;
  GSList *voronoi_region = create_voronoi_region(v);
  GSList *l;
  
  l = voronoi_region;
  while (l) {
    g_assert(one_vertex_is(GTS_TRIANGLE(l->data), v));
    area += triangle_area_exclude(GTS_TRIANGLE(l->data), v, exclude_radius);
    gts_object_destroy(GTS_OBJECT(l->data));
    l = l->next;
  }
  g_slist_free(voronoi_region);

  return (area);
}

GtsVertex *random_point_in_triangle(GtsTriangle *t)
{
  double a1 = drand48(), a2 = drand48();
  GtsVertex *v1, *v2, *v3;

  while ((a1 + a2) > 1.0) {
    a1 = drand48(); a2 = drand48();
  }

  gts_triangle_vertices(t, &v1, &v2, &v3);

  return (gts_vertex_new(gts_vertex_class(),
                         a1 * GTS_POINT(v1)->x + a2 * GTS_POINT(v2)->x + (1 - a1 - a2) * GTS_POINT(v3)->x,
                         a1 * GTS_POINT(v1)->y + a2 * GTS_POINT(v2)->y + (1 - a1 - a2) * GTS_POINT(v3)->y,
                         0.0));
}


GtsVertex *new_vertex_in_voronoi(GtsVertex *v, double exclude_radius)
{
  double area = 0.0;
  GtsVertex *vout = NULL;
  GSList *voronoi_region = create_voronoi_region(v);
  GSList *l;
  GtsTriangle *last_positive = NULL;

  l = voronoi_region;
  while (l) {
    double tarea = triangle_area_exclude(GTS_TRIANGLE(l->data), v, exclude_radius);
    area += tarea;
    if (tarea > 0.0) {
      last_positive = GTS_TRIANGLE(l->data);
    }
    l = l->next;
  }

  g_assert(last_positive != NULL);

  area *= drand48();

  l = voronoi_region;
  while (l && (area > 0.0)) {
    area -= triangle_area_exclude(GTS_TRIANGLE(l->data), v, exclude_radius);
    if (area <= 0.0) {
      vout = random_point_in_triangle_exclude(GTS_TRIANGLE(l->data), v, exclude_radius);
    }
    l = l->next;
  }

  if (! vout) {
    vout = random_point_in_triangle_exclude(last_positive, v, exclude_radius);
  }

  while (l) {
    gts_object_destroy(GTS_OBJECT(l->data));
    l = l->next;
  }
  g_slist_free(voronoi_region);

  return (vout);
}
