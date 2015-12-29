Melissa (Multi-Layer Soft Shadows)
Louis Bavoil <bavoil@sci.utah.edu>
March 4, 2007

This is a Windows demo of Melissa, a real-time soft shadow renderer
based on a multi-layer shadow map rendered from the center of a
rectangular light source. The number of layers used for the soft
shadow rendering is configurable in shaders/config.h.
See the following technical report for more information:

"Robust Soft Shadow Mapping with Depth Peeling",
Louis Bavoil, Steven P. Callahan, Claudio T. Silva,
SCI Institute Technical Report, No. UUSCI-2006-028,
University of Utah, 2006.

Controls:
---------

Drag the mouse left/middle/right buttons to move the camera 
"a/s/d/w" to move the light 
"+/-" to increase/decrease the maximum number of samples per pixel 
"r" reload the shadow shader
"f" to draw the inscribed sphere of the light and the light frustum 
