David Eberly notes:

Source code for the original paper is available online at [http://www.geometrictools.com/JGT/FastSlerp.cpp](http://www.geometrictools.com/JGT/FastSlerp.cpp) and contains the FPU-based implementation and various Intel SSE2 implementations.

The error analysis of the original paper is incorrect. A revised paper that has a correct error analysis is [https://www.geometrictools.com/Documentation/FastAndAccurateSlerp.pdf](https://www.geometrictools.com/Documentation/FastAndAccurateSlerp.pdf). In the online source code, the hard-coded constants related to the error bounds must be modified to use those mentioned in the revised PDF.