/* fast_delaunay.c - */

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

#include <math.h>
#include <stdlib.h>
#include "gts.h"

#include "weighted_tree.h"
#include "voronoi.h"

#define any_face_of(v) (GTS_FACE(GTS_EDGE((v)->segments->data)->triangles->data))

#define printf
#define g_assert(x)

double exc;

int within_01(GtsPoint *p)
{
  return ((p->x >= 0.0) &&
          (p->x <= 1.0) &&
          (p->y >= 0.0) &&
          (p->y <= 1.0));
}

int check_voronoi(gpointer item, gpointer data)
{
  GtsTriangle *t = GTS_TRIANGLE(item);
  GtsVertex *v1, *v2, *v3;
  GtsPoint *circumcenter;

  gts_triangle_vertices(t, &v1, &v2, &v3);
  
  circumcenter = gts_triangle_circumcircle_center(t, gts_point_class());

  if (within_01(circumcenter)) {
    printf("checking %f %f (%f %f %f)\n", 
           circumcenter->x, circumcenter->y,
           gts_point_distance(circumcenter, GTS_POINT(v1)),
           gts_point_distance(circumcenter, GTS_POINT(v2)),
           gts_point_distance(circumcenter, GTS_POINT(v3)));

    printf("areas %f %f %f\n",
           compute_voronoi_area(v1, exc),
           compute_voronoi_area(v2, exc),
           compute_voronoi_area(v3, exc));
    
    g_assert(gts_point_distance(circumcenter, GTS_POINT(v1)) <= exc);
    g_assert(gts_point_distance(circumcenter, GTS_POINT(v2)) <= exc);
    g_assert(gts_point_distance(circumcenter, GTS_POINT(v3)) <= exc);
  }
  gts_object_destroy(GTS_OBJECT(circumcenter));

  return 0;
}

int main (int argc, char * argv[])
{
  guint i, n;
  GtsSurface * surface;
  GSList *vertices = NULL;
  GtsTriangle * t;
  GtsVertex * v1, * v2, * v3, *first_vertex;
  GTimer * timer;
  weighted_tree *WT;
  double exclude_radius;

  if (argc != 3) {
    fprintf (stderr, "usage: random n radius\n");
    return 0;
  }

  for (i = 0; i < 5; i++) {
    drand48();
  }

  n = atoi (argv[1]);
  exclude_radius = atof(argv[2]);
  timer = g_timer_new ();

  // Create wrapping triangle
  vertices = g_slist_prepend (vertices, 
                              gts_vertex_new (gts_vertex_class (),
                                              0.0, 1.0, 0.0));
  vertices = g_slist_prepend (vertices, 
                              gts_vertex_new (gts_vertex_class (),
                                              1.0, 1.0, 0.0));
  vertices = g_slist_prepend (vertices, 
                              gts_vertex_new (gts_vertex_class (),
                                              1.0, 0.0, 0.0));
  vertices = g_slist_prepend (vertices, 
                              gts_vertex_new (gts_vertex_class (),
                                              0.0, 0.0, 0.0));

  t = gts_triangle_enclosing (gts_triangle_class (), vertices, 100.);
  gts_triangle_vertices (t, &v1, &v2, &v3);
  surface = gts_surface_new (gts_surface_class (),
			     gts_face_class (),
			     gts_edge_class (),
			     gts_vertex_class ());
  gts_surface_add_face (surface, gts_face_new (gts_face_class (),
					       t->e1, t->e2, t->e3));

  first_vertex = gts_vertex_new(gts_vertex_class(), drand48(), drand48(), 0.0);
  gts_delaunay_add_vertex(surface, first_vertex, NULL);
  
  WT = new_tree(first_vertex, compute_voronoi_area(first_vertex, exclude_radius));

  //tree_area(WT->tree, 0);
  //printf("\n\n");

  g_timer_start (timer);
  for (i = 1; i < n; i++) {
    GtsVertex *v = select_vertex_from_tree(WT);
    GtsVertex *newv = new_vertex_in_voronoi(v, exclude_radius);
    GSList *neighbors, *l;

    gts_delaunay_add_vertex(surface, newv, any_face_of(v));
    neighbors = gts_vertex_neighbors(newv, NULL, NULL);
    
    add_to_tree(WT, newv, compute_voronoi_area(newv, exclude_radius));
    
    printf("adding %f %f\n", GTS_POINT(newv)->x, GTS_POINT(newv)->y);

    l = neighbors;
    while (l) {
      printf("checking neighbor %f %f   %f\n", 
             GTS_POINT(l->data)->x, GTS_POINT(l->data)->y, 
             gts_point_distance(GTS_POINT(newv), GTS_POINT(l->data)));
      g_assert(gts_point_distance(GTS_POINT(newv), GTS_POINT(l->data)) >= exclude_radius);
      
      if (in_tree(WT, l->data))
        update_tree(WT, l->data, compute_voronoi_area(l->data, exclude_radius));
      l = l->next;
    }

    g_slist_free(neighbors);

    if ((i % 10000) == 0) {
      fprintf (stderr, "10000 points added in %f seconds (ratio %f)  total %d, %f arealeft\n", 
               g_timer_elapsed (timer, NULL),
               g_timer_elapsed(timer, NULL) / log(i), i,
               area_left(WT));
      g_timer_reset(timer);
      g_timer_start(timer);
    }

    if (no_area_left(WT)) break;

    //tree_area(WT->tree, 0);
    //      printf("\n\n");
  }

  fprintf(stderr, "Done\n");

  if (gts_delaunay_check (surface)) {
    fprintf (stderr, "WARNING: surface is not Delaunay\n");
    return 0;
  }

  print_verts(WT);

  exc = exclude_radius;
  gts_surface_foreach_face(surface, check_voronoi, NULL);
                                             
  gts_object_destroy(GTS_OBJECT(surface));
  g_node_destroy(WT->tree);
  g_hash_table_destroy(WT->table);
  g_free(WT);

  return 1;
}
