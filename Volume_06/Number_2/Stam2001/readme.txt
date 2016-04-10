A Simple Fluid Solver Based on the FFT

Jos Stam
Alias | Wavefront

This paper appears in issue Volume 6, Number 2.
Purchase this issue from the akpeters.com web site.


Abstract

This paper presents a very simple implementation of a fluid solver. The implementation is consistent with the equations of fluid flow and produces velocity fields that contain incompressible rotational structures and dynamically react to user-supplied forces. Specialized for a fluid which wraps around in space, it allows us to take advantage of the Fourier transform, which greatly simplifies many aspects of the solver. Indeed, given a Fast Fourier Transform, our solver can be implemented in roughly one page of readable C code. The solver is a good starting point for anyone interested in coding a basic fluid solver. The fluid solver presented is useful also as a basic motion primitive that can be used for many different applications in computer graphics.


Author Information

Jos Stam, Alias | Wavefront, 1218 Third Avenue, Suite 800, Seattle, WA 98101 jstam@alias.com


Addendum

[This material prepared as part of inclusion of the paper in the book Graphics Tools—The JGT Editors’ Choice.]

Since the publication of this paper the author has developed an even simpler implementation of the fluid solver which does not rely on a Fast Fourier Transform. This new solver is fully described in the following paper: Jos Stam, “Real-Time Fluid Dynamics for Games,” in Proceedings of the Game Developer Conference, March 2003.


Source Code

The C source code described in the original jgt article is available here: solver.c

The source code for the newer solver is available here: solver03.c, demo03.c


BibTeX Entry

@article{stam01,
  author = "Jos Stam",
  title = "A Simple Fluid Solver Based on the FFT",
  journal = "journal of graphics tools",
  volume = "6",
  number = "2",
  pages = "43-52",
  year = "2001",
}