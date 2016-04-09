Ray Bilinear Patch Intersections

Shaun D. Ramsey and Charles Hansen
University of Utah, School Of Computing

Kristin Potter
University of Utah

Shaun D. Ramsey and Charles Hansen
University of Utah, School of Computing

This paper appears in issue Volume 9, Number 3.
Purchase this issue from the akpeters.com web site.


Abstract

Ray tracing and other techniques employ algorithms which require the intersection between a 3D parametric ray and an object to be computed. The object to intersect is typically a sphere, triangle, or polygon but many surface types are possible. In this work we consider intersections between rays and the simplest parametric surface, the bilinear patch. Unlike other surfaces, solving the ray-bilinear patch intersection with simple algebraic manipulations fails. We present a complete, efficient, robust, and graceful formulation to solve ray-bilinear patch intersections quickly. Source code is available online.


Author Information

Shaun D. Ramsey, University of Utah, School Of Computing, 50 S. Central Campus Drive, 3190 MEB, Salt Lake City, UT 84112 ramsey@cs.utah.edu

Kristin Potter, University of Utah, School of Computing, 50 S. Central Campus Drive, 3190 MEB, Salt Lake City, UT 84112 kpotter@cs.utah.edu

Charles Hansen, University of Utah, School of Computing, 50 S. Central Campus Drive, 3190 MEB, Salt Lake City, UT 84112 hansen@cs.utah.edu


Source Code

Download the complete C++ source code and driver program in a zip archive: bilinear.zip (12K), or view the individual files:

A simple math test: main.cc
Bilinear source header: bilinear.h
Bilinear source: bilinear.cc
A vector class header: Vector.h
Vector class source: Vector.cc
The ray bilinear patch intersection software is “Open Source“ according to the MIT License.


BibTeX Entry

@article{RamseyPotterHansen04,
  author = "Shaun D. Ramsey and Kristin Potter and Charles Hansen",
  title = "Ray Bilinear Patch Intersections",
  journal = "journal of graphics, gpu, and game tools",
  volume = "9",
  number = "3",
  pages = "41-47",
  year = "2004",
}