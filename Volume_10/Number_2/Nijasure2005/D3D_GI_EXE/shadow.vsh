//-----------------------------------------------------------------------------
// Scale transformed depth in light space and send it to pixel shader
// c12      - range scale
//-----------------------------------------------------------------------------

vs.1.1

dcl_position v0
def c29, 1.0, 1.0, 1.0, 0.0

// Transform and output position
m4x4 oPos, v0, c12

// Output distance from light c20 
// light position c25
m4x4 r0, v0, c8 // world
sub r0, r0, c25 // r0 = vertex - light
dp4 r1, r0, r0
rsq r1, r1.w
rcp r0, r1 // magnitude
// scale from 1 to range => 0.0 to 1.0
sub r0, r0, c29
mul oT0.x, r0, c20

