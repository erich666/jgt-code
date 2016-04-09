Computing a View Frustum to Maximize an Object’s Image Area

Kok-Lim Low and Adrian Ilie
University of North Carolina at Chapel Hill

This paper appears in issue Volume 8, Number 1.
Purchase this issue from the akpeters.com web site.


Abstract

This paper presents a method to compute a view frustum for a 3D object viewed from a given viewpoint, such that the object is completely enclosed in the frustum and the object’s image area is also near-maximal in the given 2D rectangular viewing region. This optimization can be used to improve the resolution of shadow maps and texture maps for projective texture mapping. Instead of doing the optimization in 3D space to find a good view frustum, our method uses a 2D approach. The basic idea of our approach is as follows. First, from the given viewpoint, a conveniently-computed view frustum is used to project the 3D vertices of the object to their corresponding 2D image points. A tight 2D bounding quadrilateral is then computed to enclose these 2D image points. Next, considering the projective warp between the bounding quadrilateral and the rectangular viewing region, our method applies a technique of camera calibration to compute a new view frustum that generates an image that covers the viewing region as much as possible.


Author Information

Kok-Lim Low, Dept of Computer Science, UNC Chapel Hill, CB #3175, Sitterson Hall, Chapel Hill, NC 27599-3175 lowk@cs.unc.edu

Adrian Ilie, Dept of Computer Science, UNC Chapel Hill, CB #3175, Sitterson Hall, Chapel Hill, NC 27599-3175 adyilie@cs.unc.edu


Source Code

Downloadable C/C++ source code is available here: lowk_jgt_src.zip (63K zip archive). Several files from Numerical Recipes in C (http://numerical.recipes/) are needed as well, see the readme.txt file for details.

