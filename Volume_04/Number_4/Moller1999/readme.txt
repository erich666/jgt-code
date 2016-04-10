See http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/ for more information.

Efficiently Building a Matrix to Rotate One Vector to Another

Tomas Möller
Chalmers University of Technology

John F. Hughes
Brown University

This paper appears in issue Volume 4, Number 4.
Purchase this issue from the akpeters.com web site.


Abstract

We describe an efficient (no square roots or trigonometric functions) method to construct the 3×3 matrix that rotates a unit vector f into another unit vector t, rotating about the axis f×t. We give experimental results showing this method is faster than previously known methods. An implementation in C is provided.


Author Information

Tomas Möller, Chalmers University of Technology, Dept of Computer Engineering412 96 Gothenburg, Sweden tompa@acm.org

John F. Hughes, Brown University, Computer Science Department, 115 Waterman Street, Providence, RI 02912 jfh@cs.brown.edu


Follow-up

[May, 2003] The computation of h described on page 2 can be further optimized:

h = (1-c)/(1-c*c) = (1-c)/(v.v) = 1/(1+c)
Thanks to Gottfried Chen for pointing this out.


Source Code

The following C source file contains an implementation of the algorithm described in the paper, including the above optimization: fromtorot.c (3K HTML text)


BibTeX Entry

@article{MollerHughes99,
  author = "Tomas Möller and John F. Hughes",
  title = "Efficiently Building a Matrix to Rotate One Vector to Another",
  journal = "journal of graphics, gpu, and game tools",
  volume = "4",
  number = "4",
  pages = "1-4",
  year = "1999",
}