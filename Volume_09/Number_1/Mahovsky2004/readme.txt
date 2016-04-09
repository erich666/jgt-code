Fast Ray-Axis Aligned Bounding Box Overlap Tests with Plücker Coordinates

Jeffrey Mahovsky and Brian Wyvill
University of Calgary

This paper appears in issue Volume 9, Number 1.
Purchase this issue from the akpeters.com web site.


Abstract

Fast ray-axis aligned bounding box overlap tests can be performed by utilizing Plücker coordinates. This method tests the ray against the edges comprising the silhouette of the box instead of testing against individual faces. Projection of the edges onto a two-dimensional plane to generate the silhouette is not necessary, which simplifies the technique. The method is division-free and successive calculations are independent and consist simply of dot product operations, which permits vectorization. The method does not compute an intersection distance along the ray to the box, but this can be added as an additional step. Storage of Plücker coordinates is unnecessary, permitting integration into existing systems. Test results show the technique’s performance is up to 93% faster than traditional methods if an intersection distance is not needed.


Author Information

Jeffrey Mahovsky, University of Calgary, Dept. of Computer Science, 2500 University Drive NW, Calgary, AB T2N 1N4, Canada mahovskj@cpsc.ucalgary.edu

Brian Wyvill, University of Calgary, Department of Computer Science, 2500 University Drive NW, Calgary, AB T2N 1N4, Canada blob@cpsc.ucalgary.edu


Source Code

Sample C++ source code implementing the algorithm is available here in two versions:

Single precision: JGT-float.zip (60K zip archive)
Double precision: JGT-double.zip (60K zip archive)
In both archives the file JGT.cpp contains main() as well compilation instructions. The code has been tested under Microsoft Visual C++ .NET 2003 and Linux/g++.


BibTeX Entry

@article{MahovskyWyvill04,
  author = "Jeffrey Mahovsky and Brian Wyvill",
  title = "Fast Ray-Axis Aligned Bounding Box Overlap Tests with Plücker Coordinates",
  journal = "journal of graphics, gpu, and game tools",
  volume = "9",
  number = "1",
  pages = "35-46",
  year = "2004",
}