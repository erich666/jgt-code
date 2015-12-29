
Readme for Matlab code accompanying the paper: 
"Exact Evaluation of Catmull-Clark Subdivision Surfaces Near B-spline
Boundaries",

Script sample1.m shows the function calls necessary to make a subdivision
matrix for a given valence and face, and find its Jordan Normal Form using
subdeig().

save_eigendata(...) and save_eigendata_interior(...) generate data in a format
expected by our runtime evaluation code, and dump it to a file.  We copied
and pasted this data directly into SubdEigenData.h

See the documentation with each .m file for more information on what a
particular function does.
