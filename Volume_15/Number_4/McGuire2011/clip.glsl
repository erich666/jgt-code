/** clip.glsl
  
 SIMD optimized code to clip a triangle or quadrilateral against a plane in GLSL.

 Design goals are to:

 1. maximize coherence (to keep all threads in a warp active for scalar ALUs) by quickly reducing to a small set of common cases,
 2. maximize vector instructions (for vector ALUs)
 2. minimize peak register count (to enable a large number of simultaneous threads), and
 3. avoid non-constant array indexing (which expands to a huge set of branches on many GPUs)

*/
#line 4

const float clipEpsilon  = 0.00001;
const float clipEpsilon2 = 0.01;

/**
 Computes the intersection of triangle v0-v1-v2 with the half-space (x,y,z) * n > 0.
 The result is a convex polygon in v0-v1-v2-v3. Vertex v3 may be degenerate
 and equal to the first vertex. 

 \return number of vertices; 0, 3, or 4
*/
int clip3(const in vec3 n, in out vec3 v0, in out vec3 v1, in out vec3 v2, out vec3 v3) {

    // Distances to the plane (this is an array parallel to v[], stored as a vec3)
    vec3 dist = vec3(dot(v0, n), dot(v1, n), dot(v2, n));

    if (! any(greaterThanEqual(dist, vec3(clipEpsilon2)))) {
        // All clipped
        return 0;
    }  
    
    if (all(greaterThanEqual(dist, vec3(-clipEpsilon)))) {
        // None clipped (original triangle vertices are unmodified)
        v3 = v0;
        return 3;

    }
        
    bvec3 above = greaterThanEqual(dist, vec3(0.0));

    // There are either 1 or 2 vertices above the clipping plane.
    bool nextIsAbove;

    // Find the CCW-most vertex above the plane by cycling
    // the vertices in place.  There are three cases.
    if (above[1] && ! above[0]) {
        nextIsAbove = above[2];
        // Cycle once CCW.  Use v3 as a temp
        v3 = v0; v0 = v1; v1 = v2; v2 = v3;
        dist = dist.yzx;
    } else if (above[2] && ! above[1]) {
        // Cycle once CW.  Use v3 as a temp.
        nextIsAbove = above[0];
        v3 = v2; v2 = v1; v1 = v0; v0 = v3;
        dist = dist.zxy;
    } else {
        nextIsAbove = above[1];
    }
    // Note: The above[] values are no longer in sync with v values and dist[].

    // We always need to clip v2-v0.
    v3 = mix(v0, v2, dist[0] / (dist[0] - dist[2]));

    if (nextIsAbove) {

        // There is a quadrilateral above the plane
        //
        //    v0---------v1
        //      \        |
        //   ....v3......v2'...
        //          \    |
        //            \  |
        //              v2

        v2 = mix(v1, v2, dist[1] / (dist[1] - dist[2]));
        return 4;
    } else {

        // There is a triangle above the plane
        //
        //            v0
        //           / |
        //         /   |
        //   ....v2'..v1'...
        //      /      |
        //    v2-------v1

        v1 = mix(v0, v1, dist[0] / (dist[0] - dist[1]));
        v2 = v3;
        v3 = v0;
        return 3;
    }
}


/**
 Computes the intersection of quadrilateral v0-v1-v2-v3 with the half-space (x,y,z)*n > 0.  
 If there is no intersection, returns 0. If there is an intersection,
 returns the number of unique vertices, k.  Vertex [(k+1) % 5] is always equal to v0.

 \return number of vertices; 0, 3, 4, or 5
 */
int clip4(const in vec3 n, in out vec3 v0, in out vec3 v1, in out vec3 v2, in out vec3 v3, out vec3 v4) {
    // Distances to the plane (this is an array parallel to v[], stored as a vec4)
    vec4 dist = vec4(dot(v0, n), dot(v1, n), dot(v2, n), dot(v3, n));

    const float epsilon = 0.00001;

    if (! any(greaterThanEqual(dist, vec4(clipEpsilon2)))) {
        // All clipped;
        return 0;
    } 
    
    if (all(greaterThanEqual(dist, vec4(-clipEpsilon)))) {
        // None clipped (original quad vertices are unmodified)
        v4 = v0;
        return 4;
    }
    
    // There are exactly 1, 2, or 3 vertices above the clipping plane.

    bvec4 above = greaterThanEqual(dist, vec4(0.0));

    // Make v0 the ccw-most vertex above the plane by cycling
    // the vertices in place.  There are four cases.
    if (above[1] && ! above[0]) {
        // v1 is the CCW-most, so cycle values CCW
        // using v4 as a temp.
        v4 = v0; v0 = v1; v1 = v2; v2 = v3; v3 = v4;
        dist = dist.yzwx;
    } else if (above[2] && ! above[1]) {
        // v2 is the CCW-most. Cycle twice CW using v4 as a temp, i.e., swap v0 with v2 and v3 with v1.
        v4 = v0; v0 = v2; v2 = v4;
        v4 = v1; v1 = v3; v3 = v4;
        dist = dist.zwxy;
    } else if (above[3] && ! above[2]) {
        // v3 is the CCW-most, so cycle values CW using v4 as a temp
        v4 = v0; v0 = v3; v3 = v2; v2 = v1; v1 = v4;
        dist = dist.wxyz;
    }

    // Note: The above[] values are no longer in sync with v values and and dist[].

    // We now need to clip along edge v3-v0 and one of edge v0-v1, v1-v2, or v2-v3.
    // Since we *always* have to clip v3-v0, compute that first and store the result in v4.
    v4 = mix(v0, v3, dist[0] / (dist[0] - dist[3]));

    int numAbove = int(above[0]) + int(above[1]) + int(above[2]) + int(above[3]);
    switch (numAbove) {
    case 1:
        // Clip v0-v1, output a triangle
        //
        //            v0
        //           / |
        //         /   |
        //   ...v3'....v1'...
        //      /      |
        //    v3--v2---v1

        v1 = mix(v0, v1, dist[0] / (dist[0] - dist[1]));
        v2 = v4;
        v3 = v4 = v0;
        return 3;

    case 2:
        // Clip v1-v2, output a quadrilateral
        //
        //    v0-----------v1
        //      \           |
        //   ....v3'...... v2'...
        //          \       |
        //            v3---v2
        //              

        v2 = mix(v1, v2, dist[1] / (dist[1] - dist[2]));
        v3 = v4;
        v4 = v0;
        return 4;

    case 3:
        // Clip v2-v3, output a pentagon
        //
        //    v0----v1----v2
        //      \        |
        //   .....v4....v3'...
        //          \   |
        //            v3
        //              
        v3 = mix(v2, v3, dist[2] / (dist[2] - dist[3]));
        return 5;
    } // switch
} 
