vs.2.0
dcl_position v0
dcl_normal v3
dcl_texcoord v7

def c29, 1.0, 1.0, 1.0, 1.0
def c30, 0.0055, 0.0, 0.0, 0.0 //0.0055

// Pass texture coordinates to pixel shader.
mov oT0, v7

// Transform and project vertex for eye.
m4x4 oPos, v0, c12

// Create texture coordinates for shadow map.
m4x4 r0, v0, c8			// Object Space to World Space
m4x4 r1, r0, c16		// World Space to shadow map texture coordinates.
mov oT1, r1

// Pass world space position.
mov oT2, r0

// Pass volume map texture coordinates. => t5
sub r1, r0, c31
mul oT5, r1, c32

// Pass distance from light for shadowmap computation.
sub r0, r0, c25 // r0 = vertex position - light position
dp4 r1, r0, r0
rsq r1, r1.w
rcp r0, r1.x // magnitude
sub r0, r0, c29 // scale 1 to range => 0.0 to 1.0
mul r0, r0, c20.x
sub oT4, r0, c30.x

// Pass normal
mov oT3, v3
