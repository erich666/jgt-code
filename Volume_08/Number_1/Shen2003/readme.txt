A Fast Triangle-Triangle Overlap Test Using Signed Distances

Hao Shen and Zesheng Tang
Tsinghua University

Pheng Ann Heng
The Chinese University of Hong Kong

This paper appears in issue Volume 8, Number 1.
Purchase this issue from the akpeters.com web site.


Abstract

A fast test for triangle-triangle intersection by computing signed vertex-plane distances (sufficient if one triangle is wholly to one side of the other) and sign line-line distances of selected edges (otherwise) is presented. This algorithm is faster than previously published algorithms and the code is available online.


Author Information

Hao Shen, Dept of Computer Science & Technology, Tsinghua University100084 Beijing, China hshen@vis.cs.tsinghua.edu.cn

Pheng Ann Heng, The Chinese University of Hong Kong, Dept of Computer Science & Engineering Shatin, Hong Kong pheng@cse.cuhk.edu.hk

Zesheng Tang, Dept of Computer Science & Technology, Tsinghua University100084 Beijing, China ztang@must.edu.mo


Editor’s Note

This is one of two simultaneous triangle-triangle overlap papers in this issue. See also Guigue and Devillers 03


Source Code

C source code of our triangle-triangle intersection algorithm is available here: tri_tri.c (12K HTML text).

A test package, built using Microsoft Visual C++, is available here: tri_tri_test.zip (770K zip archive). In addition to C source code, this package also include the data set generator and testing data sets. Performance can be tested on the test data sets presented in the package. If you don’t want to use the presented data sets, use the generator to build new test data sets before testing. After adjusting the parameters defined in “./include/datatype.h” and rebuilding the “Performance” project, run “./bin/Performance.exe“ to watch the cost time of our algorithm or Möller's under different intersection ratio.

