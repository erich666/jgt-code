/* triangle_area.c - */

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

#include <math.h>
#include <stdlib.h>
#include "gts.h"

#include "triangle_area.h"
#define printf
#define g_assert(x) 

double sgn(double a)
{
  if (a < 0.0) return (-1.0);
  return (1.0);
}

int between(double a, double b, double c)
{
  return (((a <= b) && (b <= c)) ||
          ((a >= b) && (b >= c)));
}

void clip_to_circle_origin(double r, double x1, double y1, double x2, double y2,
                           double *ix1, double *iy1, double *ix2, double *iy2)
{
  double dx, dy, dr2, D;

  // from MathWorld

  dx = x2 - x1;
  dy = y2 - y1;
  dr2 = dx*dx + dy*dy;
  D = x1 * y2 - x2 * y1;

  if ((r*r*dr2-D*D) <= 0.0) {
    // no intersection?
    fprintf(stderr, "WARNING: no intersection in clip_to_circle_origin\n");
    *ix1 = x2; *iy1 = y2;
    return;
  }

  *ix1 = (D*dy + sgn(dy)*dx*sqrt(r*r*dr2-D*D))/dr2;
  *iy1 = (-D*dx + fabs(dy)*sqrt(r*r*dr2-D*D))/dr2;
  
  *ix2 = (D*dy - sgn(dy)*dx*sqrt(r*r*dr2-D*D))/dr2;
  *iy2 = (-D*dx - fabs(dy)*sqrt(r*r*dr2-D*D))/dr2;

  return;
}

#define SQR(a) ((a)*(a))

GtsVertex *clip_to_circle(GtsVertex *center, double exclude_radius, GtsVertex *out, GtsVertex *in)
{
  double x1, y1, x2, y2;
  clip_to_circle_origin(exclude_radius, 
                        GTS_POINT(out)->x - GTS_POINT(center)->x, GTS_POINT(out)->y - GTS_POINT(center)->y,
                        GTS_POINT(in)->x - GTS_POINT(center)->x, GTS_POINT(in)->y - GTS_POINT(center)->y,
                        &x1, &y1, &x2, &y2);

  x1 += GTS_POINT(center)->x;
  y1 += GTS_POINT(center)->y;
  x2 += GTS_POINT(center)->x;
  y2 += GTS_POINT(center)->y;

  printf("clipping %f %f     %f %f   cetner %f %f\n     %f %f      %f %f\n",
         GTS_POINT(out)->x, GTS_POINT(out)->y, 
         GTS_POINT(in)->x, GTS_POINT(in)->y, 
         GTS_POINT(center)->x, GTS_POINT(center)->y,
         x1, y1, x2, y2);
         

  if ((SQR(x1 - GTS_POINT(out)->x) + SQR(y1 - GTS_POINT(out)->y)) <
      (SQR(x2 - GTS_POINT(out)->x) + SQR(y2 - GTS_POINT(out)->y))) {

    return (gts_vertex_new(gts_vertex_class(), x1, y1, 0.0));
  } else {
    return (gts_vertex_new(gts_vertex_class(), x2, y2, 0.0));
  }
}

void closest_approach_circle_origin(double x1, double y1, double x2, double y2,
                                    double *ix, double *iy)
{
  double dx, dy, dr2, D;

  // same as intersect above, but assume that discriminant is zero

  dx = x2 - x1;
  dy = y2 - y1;
  dr2 = dx*dx + dy*dy;
  D = x1 * y2 - x2 * y1;

  *ix = (D*dy)/dr2;
  *iy = (-D*dx)/dr2;

  if (! ((between(x1, *ix, x2) && (between(y1, *iy, y2))))) {
    if ((SQR(x1-*ix)+SQR(y1-*iy)) < (SQR(x2-*ix)+SQR(y2-*iy))) {
      *ix = x1;
      *iy = y1;
    } else {
      *ix = x2;
      *iy = y2;
    }
  }

  return;
}


GtsVertex *closest_approach_circle(GtsVertex *center, GtsVertex *v1, GtsVertex *v2)
{
  double x, y;
  closest_approach_circle_origin(GTS_POINT(v1)->x - GTS_POINT(center)->x, GTS_POINT(v1)->y - GTS_POINT(center)->y,
                                 GTS_POINT(v2)->x - GTS_POINT(center)->x, GTS_POINT(v2)->y - GTS_POINT(center)->y,
                                 &x, &y);

  return (gts_vertex_new(gts_vertex_class(), 
                         x + GTS_POINT(center)->x,
                         y + GTS_POINT(center)->y,
                         0.0));
}

void clip_to_circle_both(GtsVertex *center, double exclude_radius, GtsVertex *v1, GtsVertex *v2, GtsVertex **out1, GtsVertex **out2)
{
  GtsVertex *closest_approach = closest_approach_circle(center, v1, v2);

  g_assert(gts_point_distance(GTS_POINT(center), GTS_POINT(closest_approach)) < exclude_radius);

  *out1 = clip_to_circle(center, exclude_radius, v1, closest_approach);
  *out2 = clip_to_circle(center, exclude_radius, v2, closest_approach);
  
  gts_object_destroy(GTS_OBJECT(closest_approach));
}


int intersects_circle(GtsVertex *center, double exclude_radius, GtsVertex *v1, GtsVertex *v2)
{
  GtsVertex *closest_approach = closest_approach_circle(center, v1, v2);

  if (gts_point_distance(GTS_POINT(center), GTS_POINT(closest_approach)) < exclude_radius) {
    gts_object_destroy(GTS_OBJECT(closest_approach));
    return (1);
  } else {
    gts_object_destroy(GTS_OBJECT(closest_approach));
    return (0);
  }
}

double threepts_area_exclude_compute(GtsVertex *v1, GtsVertex *v2, GtsVertex *v3, double exclude_radius)
{
  GtsVector v12, v13;
  double triarea, angle, circarea, z;

  gts_vector_init(v12, GTS_POINT(v1), GTS_POINT(v2));
  gts_vector_init(v13, GTS_POINT(v1), GTS_POINT(v3));

  z = fabs(v12[0] * v13[1] - v12[1] * v13[0]);
  triarea = z / 2.0;
  
  if (z == 0.0) {
    angle = 0.0;
  } else {
    angle = asin(z / (gts_vector_norm(v12) * gts_vector_norm(v13)));
  }

  circarea = 0.5 * angle * exclude_radius * exclude_radius;

  if ((triarea - circarea) <= 0.0) {
    return (0.0);
  } else {
    return (triarea - circarea);
  }
}

double threepts_area_exclude(GtsVertex *v1, GtsVertex *v2, GtsVertex *v3, double exclude_radius)
{
  int intersects12 = (gts_point_distance(GTS_POINT(v1), GTS_POINT(v2)) > exclude_radius);
  int intersects13 = (gts_point_distance(GTS_POINT(v1), GTS_POINT(v3)) > exclude_radius);


  // easy case
  if ((! intersects12) & (! intersects13)) return (0.0);

  // one vertex out
  if ((intersects12) & (! intersects13)) {
    GtsVertex *v23 = clip_to_circle(v1, exclude_radius, v2, v3);
    double area = threepts_area_exclude_compute(v1, v2, v23, exclude_radius);
    gts_object_destroy(GTS_OBJECT(v23));
    return area;
  }

  // one vertex out
  if ((! intersects12) & (intersects13)) {
    GtsVertex *v32 = clip_to_circle(v1, exclude_radius, v3, v2);
    double area = threepts_area_exclude_compute(v1, v32, v3, exclude_radius);
    gts_object_destroy(GTS_OBJECT(v32));
    return area;
  }

  // two vertices out
  if (intersects_circle(v1, exclude_radius, v2, v3)) {
    // two separate areas
    GtsVertex *v23_near2, *v32_near3;
    double area21, area31;

    clip_to_circle_both(v1, exclude_radius, v2, v3, &v23_near2, &v32_near3);

    area21 = threepts_area_exclude_compute(v1, v2, v23_near2, exclude_radius);
    area31 = threepts_area_exclude_compute(v1, v3, v32_near3, exclude_radius);
    
    gts_object_destroy(GTS_OBJECT(v23_near2));
    gts_object_destroy(GTS_OBJECT(v32_near3));
    return (area21 + area31);
  } else {
    // one area
    return (threepts_area_exclude_compute(v1, v2, v3, exclude_radius));
  }
}

double threepts_area(GtsVertex *v1, GtsVertex *v2, GtsVertex *v3)
{
  GtsVector v12, v13;
  double z;

  gts_vector_init(v12, GTS_POINT(v1), GTS_POINT(v2));
  gts_vector_init(v13, GTS_POINT(v1), GTS_POINT(v3));

  z = fabs(v12[0] * v13[1] - v12[1] * v13[0]);

  return (z / 2.0);
}


static int same_point(GtsVertex *v1, GtsVertex *v2)
{
  return ((GTS_POINT(v1)->x == GTS_POINT(v2)->x) &&
          (GTS_POINT(v1)->y == GTS_POINT(v2)->y));
}


double triangle_area_exclude(GtsTriangle *t, GtsVertex *v, double exclude_radius)
{
  GtsVertex *v1, *v2, *v3;
  
  gts_triangle_vertices(t, &v1, &v2, &v3);
  if (same_point(v, v2)) {
    v2 = v1;
    v1 = v;
  }
  else if (same_point(v, v3)) {
    v3 = v1;
    v1 = v;
  }

  g_assert(same_point(v, v1));
    
  return (threepts_area_exclude(v1, v2, v3, exclude_radius));
}



void random_point_in_threepts(GtsVertex *v1, GtsVertex *v2, GtsVertex *v3, GtsVertex *out)
{
  double a1 = drand48(), a2 = drand48();
  
  while ((a1 + a2) > 1.0) {
    a1 = drand48(); a2 = drand48();
  }
  
  printf("a1, a2, %f %f\n", a1, a2);

  printf("source: %f %f    %f %f    %f %f\n",
         GTS_POINT(v1)->x, GTS_POINT(v1)->y, 
         GTS_POINT(v2)->x, GTS_POINT(v2)->y, 
         GTS_POINT(v3)->x, GTS_POINT(v3)->y);

  gts_point_set(GTS_POINT(out),
                a1 * GTS_POINT(v1)->x + a2 * GTS_POINT(v2)->x + (1 - a1 - a2) * GTS_POINT(v3)->x,
                a1 * GTS_POINT(v1)->y + a2 * GTS_POINT(v2)->y + (1 - a1 - a2) * GTS_POINT(v3)->y,
                0.0);
}


#define MAXCOUNT 100000
GtsVertex *random_point_in_threepts_exclude_compute(GtsVertex *center, GtsVertex *v1, GtsVertex *v2, GtsVertex *v3, double exclude_radius)
{
  int count = 0;
  GtsVertex *out = gts_vertex_new(gts_vertex_class(), 0.0, 0.0, 0.0);


  printf("dists: %e %e %e\n",
         gts_point_distance(GTS_POINT(v1), GTS_POINT(center)) - exclude_radius,
         gts_point_distance(GTS_POINT(v2), GTS_POINT(center)) - exclude_radius,
         gts_point_distance(GTS_POINT(v3), GTS_POINT(center)) - exclude_radius);

  g_assert(gts_point_distance(GTS_POINT(v1), GTS_POINT(center)) >= (exclude_radius - 1e-15));
  g_assert(gts_point_distance(GTS_POINT(v2), GTS_POINT(center)) >= (exclude_radius - 1e-15));
  g_assert(gts_point_distance(GTS_POINT(v3), GTS_POINT(center)) >= (exclude_radius - 1e-15));



  for (count = 0; count < MAXCOUNT; count++) {
  printf("source: %f %f    %f %f    %f %f    (exc %f %f, %f)\n",
         GTS_POINT(v1)->x, GTS_POINT(v1)->y, 
         GTS_POINT(v2)->x, GTS_POINT(v2)->y, 
         GTS_POINT(v3)->x, GTS_POINT(v3)->y,
         GTS_POINT(center)->x, GTS_POINT(center)->y, exclude_radius);

  printf("    Dists: %f %f %f\n",
         gts_point_distance(GTS_POINT(v1), GTS_POINT(center)),
         gts_point_distance(GTS_POINT(v2), GTS_POINT(center)),
         gts_point_distance(GTS_POINT(v3), GTS_POINT(center)));



    random_point_in_threepts(v1, v2, v3, out);

    if (gts_point_distance(GTS_POINT(center), GTS_POINT(out)) >= exclude_radius)
      return (out);
  }

  fprintf(stderr, "Died\n");

  exit(-1);

  return (out);
}



GtsVertex *random_point_in_threepts_exclude(GtsVertex *v1, GtsVertex *v2, GtsVertex *v3, double exclude_radius)
{
  int intersects12 = (gts_point_distance(GTS_POINT(v1), GTS_POINT(v2)) > exclude_radius);
  int intersects13 = (gts_point_distance(GTS_POINT(v1), GTS_POINT(v3)) > exclude_radius);

  // one edge of triangle must be outside
  g_assert(intersects12 || intersects13);

  // one vertex out -> v2
  if ((intersects12) & (! intersects13)) {
    GtsVertex *v21 = clip_to_circle(v1, exclude_radius, v2, v1);
    GtsVertex *v23 = clip_to_circle(v1, exclude_radius, v2, v3);
    GtsVertex *out = random_point_in_threepts_exclude_compute(v1, v2, v21, v23, exclude_radius);
    printf ("v2 out\n");
    gts_object_destroy(GTS_OBJECT(v21));
    gts_object_destroy(GTS_OBJECT(v23));
    return out;
  }

  // one vertex out -> v3
  if ((! intersects12) & (intersects13)) {
    GtsVertex *v31 = clip_to_circle(v1, exclude_radius, v3, v1);
    GtsVertex *v32 = clip_to_circle(v1, exclude_radius, v3, v2);
    GtsVertex *out = random_point_in_threepts_exclude_compute(v1, v3, v31, v32, exclude_radius);
    printf ("v3 out\n");

    gts_object_destroy(GTS_OBJECT(v31));
    gts_object_destroy(GTS_OBJECT(v32));
    return out;
  }

  // two vertices out 
  if (intersects_circle(v1, exclude_radius, v2, v3)) {
    // two separate areas
    GtsVertex *out;
    GtsVertex *v23_near2, *v32_near3;
    double area21, area31;

    printf ("both out - intersecting\n");

    clip_to_circle_both(v1, exclude_radius, v2, v3, &v23_near2, &v32_near3);

    area21 = threepts_area_exclude_compute(v1, v2, v23_near2, exclude_radius);
    area31 = threepts_area_exclude_compute(v1, v3, v32_near3, exclude_radius);
    if ((drand48() * (area21 + area31)) > area31) {
      GtsVertex *v21 = clip_to_circle(v1, exclude_radius, v2, v1);
      out = random_point_in_threepts_exclude_compute(v1, v21, v23_near2, v2, exclude_radius);
      gts_object_destroy(GTS_OBJECT(v21));
    } else {
      GtsVertex *v31 = clip_to_circle(v1, exclude_radius, v3, v1);
      out = random_point_in_threepts_exclude_compute(v1, v31, v32_near3, v3, exclude_radius);
      gts_object_destroy(GTS_OBJECT(v31));
    }
    
    gts_object_destroy(GTS_OBJECT(v23_near2));
    gts_object_destroy(GTS_OBJECT(v32_near3));
    return (out);
  }

  // final case, v2->v3 doesn't intersect circle
  {
    GtsVertex *out;
    GtsVertex *closest_approach = closest_approach_circle(v1, v2, v3);
    double area12c = threepts_area_exclude_compute(v1, v2, closest_approach, exclude_radius);
    double area13c = threepts_area_exclude_compute(v1, v3, closest_approach, exclude_radius);
    printf ("both out - non-intersecting\n");

    if ((drand48() * (area12c + area13c)) > area13c) {
      // select from triangle 1,2,closest_approach (with exclusion)
      GtsVertex *vc1 = clip_to_circle(v1, exclude_radius, closest_approach, v1);
      double area_all_out = threepts_area(v2, vc1, closest_approach);
      if ((drand48() * area12c) > area_all_out) {
        GtsVertex *v21 = clip_to_circle(v1, exclude_radius, v2, v1);
        // select from triangle v21, vc1, v2 (with exclusion)
        printf("case 1\n");
        printf("vc1 dist %f\n", gts_point_distance(GTS_POINT(vc1), GTS_POINT(v1)));
        printf("v2 dist %f\n", gts_point_distance(GTS_POINT(v2), GTS_POINT(v1)));
        printf("v21 dist %f\n", gts_point_distance(GTS_POINT(v21), GTS_POINT(v1)));

        printf("center %f %f\n", GTS_POINT(v1)->x, GTS_POINT(v1)->y);
        printf("vc1 %f %f\n", GTS_POINT(vc1)->x, GTS_POINT(vc1)->y);
        printf("v2 %f %f\n", GTS_POINT(v2)->x, GTS_POINT(v2)->y);
        printf("v21 %f %f\n", GTS_POINT(v21)->x, GTS_POINT(v21)->y);
        printf("closest_approach %f %f\n", GTS_POINT(closest_approach)->x, GTS_POINT(closest_approach)->y);

        out = random_point_in_threepts_exclude_compute(v1, v21, vc1, v2, exclude_radius);
        printf("out %f %f\n", GTS_POINT(out)->x, GTS_POINT(out)->y);


        g_assert(gts_point_distance(GTS_POINT(v1), GTS_POINT(out)) >= exclude_radius);
        gts_object_destroy(GTS_OBJECT(v21));
      } else {
        // select from triangle v2, vc1, closest_approach (no exclusion)
        out = gts_vertex_new(gts_vertex_class(), 0.0, 0.0, 0.0);
        printf("case 2\n");
        printf("vc1 dist %f\n", gts_point_distance(GTS_POINT(vc1), GTS_POINT(v1)));
        printf("v2 dist %f\n", gts_point_distance(GTS_POINT(v2), GTS_POINT(v1)));
        printf("closest_approach dist %f (should be >= %f)\n", gts_point_distance(GTS_POINT(closest_approach), GTS_POINT(v1)), exclude_radius);
        printf("center %f %f\n", GTS_POINT(v1)->x, GTS_POINT(v1)->y);
        printf("vc1 %f %f\n", GTS_POINT(vc1)->x, GTS_POINT(vc1)->y);
        printf("v2 %f %f\n", GTS_POINT(v2)->x, GTS_POINT(v2)->y);
        printf("closest_approach %f %f\n", GTS_POINT(closest_approach)->x, GTS_POINT(closest_approach)->y);

        random_point_in_threepts(vc1, v2, closest_approach, out);
        printf("out %f %f\n", GTS_POINT(out)->x, GTS_POINT(out)->y);


        g_assert(gts_point_distance(GTS_POINT(v1), GTS_POINT(out)) >= exclude_radius);

      }
      gts_object_destroy(GTS_OBJECT(vc1));
    } else {
      // select from triangle 1,3,closest_approach (with exclusion)
      GtsVertex *vc1 = clip_to_circle(v1, exclude_radius, closest_approach, v1);
      double area_all_out = threepts_area(v3, vc1, closest_approach);
      if ((drand48() * area13c) > area_all_out) {
        // select from triangle v31, vc1, v3 (with exclusion)
        printf("case 3\n");
        GtsVertex *v31 = clip_to_circle(v1, exclude_radius, v3, v1);

        printf("center %f %f\n", GTS_POINT(v1)->x, GTS_POINT(v1)->y);
        printf("vc1 %f %f\n", GTS_POINT(vc1)->x, GTS_POINT(vc1)->y);
        printf("v3 %f %f\n", GTS_POINT(v3)->x, GTS_POINT(v3)->y);
        printf("v31 %f %f\n", GTS_POINT(v31)->x, GTS_POINT(v31)->y);
        printf("closest_approach %f %f\n", GTS_POINT(closest_approach)->x, GTS_POINT(closest_approach)->y);


        printf("vc1 dist %f %e\n", gts_point_distance(GTS_POINT(vc1), GTS_POINT(v1)), exclude_radius - gts_point_distance(GTS_POINT(vc1), GTS_POINT(v1)));
        printf("v31 dist %f\n", gts_point_distance(GTS_POINT(v31), GTS_POINT(v1)));
        printf("v3 dist %f\n", gts_point_distance(GTS_POINT(v3), GTS_POINT(v1)));
        printf("closest_approach dist %f (should be >= %f)\n", gts_point_distance(GTS_POINT(closest_approach), GTS_POINT(v1)), exclude_radius);

        g_assert(gts_point_distance(GTS_POINT(v3), GTS_POINT(v1)) >= exclude_radius);

        out = random_point_in_threepts_exclude_compute(v1, v31, v3, vc1, exclude_radius);
        gts_object_destroy(GTS_OBJECT(v31));
      } else {
        // select from triangle v3, vc1, closest_approach (no exclusion)
        out = gts_vertex_new(gts_vertex_class(), 0.0, 0.0, 0.0);
        printf("case 4\n");
        printf("closest_approach %f %f\n", GTS_POINT(closest_approach)->x, GTS_POINT(closest_approach)->y);

        printf("vc1 %f %f\n", GTS_POINT(vc1)->x, GTS_POINT(vc1)->y);

        random_point_in_threepts(v3, vc1, closest_approach, out);
        printf("out %f %f\n", GTS_POINT(out)->x, GTS_POINT(out)->y);
      }
      gts_object_destroy(GTS_OBJECT(vc1));
    }
    gts_object_destroy(GTS_OBJECT(closest_approach));
    return (out);
  }
}


GtsVertex *random_point_in_triangle_exclude(GtsTriangle *t, GtsVertex *v, double exclude_radius)
{
  GtsVertex *v1, *v2, *v3, *vout;
  
  gts_triangle_vertices(t, &v1, &v2, &v3);
  if (same_point(v, v2)) {
    v2 = v1;
    v1 = v;
  }
  else if (same_point(v, v3)) {
    v3 = v1;
    v1 = v;
  }

  printf("TAKING from %f %f \n   %f %f \n    %f %f\n",
         GTS_POINT(v1)->x, GTS_POINT(v1)->y, 
         GTS_POINT(v2)->x, GTS_POINT(v2)->y, 
         GTS_POINT(v3)->x, GTS_POINT(v3)->y);

  g_assert(same_point(v, v1));
    
  g_assert((gts_point_distance(GTS_POINT(v1), GTS_POINT(v2)) >= exclude_radius) ||
           (gts_point_distance(GTS_POINT(v1), GTS_POINT(v3)) >= exclude_radius));

  vout = random_point_in_threepts_exclude(v1, v2, v3, exclude_radius);

  printf("dist: %f\n", gts_point_distance(GTS_POINT(v1), GTS_POINT(vout)));
  g_assert(gts_point_distance(GTS_POINT(v1), GTS_POINT(vout)) >= exclude_radius);

  return (vout);
}
