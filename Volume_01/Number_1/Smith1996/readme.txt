Source Code

Downloadable C code that implements the color transforms described in the paper: HWB to RGB transforms, and the closely related HSV to RGB transforms.

Revision history:

31 Jan 1996
Initial version.
28 Mar 1997
HSV to RGB: Corrected bug in test if i is even (see erratum below).
27 Oct 1997
HSV to RGB: deleted spurious line b = 1-v;
Errata

On page 17, in the source code for the routine HSV_to_RGB(), the line

    if (i^1) f = 1 - f; // if i is even
should be:

    if (!(i&1)) f = 1 - f; // if i is even

This correction has been made in the downloadable source code listed above.