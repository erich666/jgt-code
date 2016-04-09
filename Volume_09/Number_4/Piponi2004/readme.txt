Automatic Differentiation, C++ Templates, and Photogrammetry

Dan Piponi
ESC Entertainment

This paper appears in issue Volume 9, Number 4.
Purchase this issue from the akpeters.com web site.


Abstract

Differential calculus is ubiquitous in digital movie production. We give a novel presentation of automatic differentiation, a method for computing derivatives of functions, that is not well known within the graphics community, and describe some applications of this method. In particular we describe the implementation of a photogrammetric reconstruction tool used on the post-production of Matrix Reloaded and Matrix Revolutions that was built using automatic differentiation.


Author Information

Dan Piponi, 877 Walavista Avenue, Oakland, CA 94610 d.piponi@sigfpe.com


Source Code

This file contains sample C++ source code illustrating the methods in the paper: Adiff-1.0.tgz (4K gzipped tar archive)


Errata

On page 46, line 4, the term f(X(3.0f)+d) should be:

    f(Dual<X>(3.0f)+d)
	

BibTeX Entry

@article{Piponi04,
  author = "Dan Piponi",
  title = "Automatic Differentiation, C++ Templates, and Photogrammetry",
  journal = "journal of graphics, gpu, and game tools",
  volume = "9",
  number = "4",
  pages = "41-55",
  year = "2004",
}