// This code is by Peter Shirley and is in the public domain.
// It maps points in [0,1)^2 to the unit disk centered at the origin.
// The algorithm is described in A Low Distortion Map Between Disk and Square 
// by Shirley and Chiu 2(3).
// The code below is much nicer than the code in the original article and is 
// a slightly modification of code by David Cline. It includes two small
// improvements suggested by Franz and by Greg Ward in a blog discussion
// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// classic polar method
void classic_square_to_disk(float x_in, float y_in, float *x_out, float *y_out) {
    float r = sqrt(x_in);
    float theta = 2*M_PI*y_in;
    *x_out = r*cos(theta);
    *y_out = r*sin(theta);
}
 
// low distortion method
void lowd_square_to_disk(float x_in, float y_in, float *x_out, float *y_out) {
    float theta,r;
    float a = 2*x_in - 1;
    float b = 2*y_in - 1;
    if (a == 0 && b == 0) {
        r = theta = 0;
    }
    else if (a*a> b*b) { 
        r = a;
        theta = (M_PI/4)*(b/a);
    } else {
        r = b;
        theta = (M_PI/2) - (M_PI/4)*(a/b);
    }
    *x_out = r*cos(theta);
    *y_out = r*sin(theta);
}


// take a set of 50^2 jittered samples to the disk.
int main() {
    float x, y;
    float x_in, y_in;
    int sqrt_n = 50;
    for (int i = 0; i < sqrt_n; i++) {
       for (int j = 0; j < sqrt_n; j++) {
          x_in = (i + drand48()) / sqrt_n;
          y_in = (j + drand48()) / sqrt_n;
          lowd_square_to_disk(x_in, y_in, &x, &y);
          printf("%lf,%lf\n", x, y);
       }
    }
}
