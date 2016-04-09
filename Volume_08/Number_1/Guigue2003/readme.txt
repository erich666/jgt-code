From https://web.archive.org/web/20120118225810/http://jgt.akpeters.com/papers/GuigueDevillers03/

Fast and Robust Triangle-Triangle Overlap Test using Orientation Predicates

Philippe Guigue and Olivier Devillers
INRIA Sophia-Antipolis

This paper appears in issue Volume 8, Number 1.
Purchase this issue from the akpeters.com web site.


Abstract

This paper presents an algorithm for determining whether two triangles in three dimensions intersect. The general scheme is identical to the one proposed by Möller. The main difference is that our algorithm relies exclusively on the sign of 4 x 4 determinants and does not need any intermediate explicit constructions which are source of numerical errors. Besides the fact that the resulting code is more reliable than existing methods, it is also more efficient. The source code is available online.

Author Information

Philippe Guigue, INRIA Sophia-Antipolis, BP 93, 2004 Route des Lucioles06902 Sophia-Antipolis Cedex, France Philippe.Guigue@sophia.inria.fr

Olivier Devillers, INRIA Sophia-Antipolis, BP 93, 2004 Route des Lucioles06902 Sophia-Antipolis Cedex, France Olivier.Devillers@sophia.inria.fr


Editor’s Note

This is one of two simultaneous triangle-triangle overlap papers in this issue; see also Shen, Heng and Tang 03. This paper has also been chosen for inclusion in the book Graphics Tools—The JGT Editors’ Choice; see addendum below.


Source Code

Downloadable C source code is available here: triangle_triangle_intersection.c (19K HTML text). It contains three routines:

tri_tri_overlap_test_3d(), the three-dimensional predicate described in the paper

tri_tri_overlap_test_2d(), the two-dimensional predicate

tri_tri_intersection_test_3d(), the version that computes the line segment of intersection if the triangles do overlap and are not coplanar.


Revision history

July 2002
Program creation.
October 2002
Added optimizations.
January 2003
Added version that computes the line of intersection.
June 2003
Bug fix. Thanks to Tushar Udeshi for pointing it out!
December 2003
Bug fix.
Addendum


A Ray-Triangle Intersection Test

As part of the inclusion of this paper in the book Graphics Tools—The JGT Editors’ Choice, the authors discuss how the value of determinants introduced in the paper can be used to obtain an efficient solution for the ray-triangle intersection problem, and provide source code. See: Fast Ray-Triangle Intersection Test Using Orientation Determinants

