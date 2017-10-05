/*
 * display_mac.c
 *
 *	Machine specific code to display pixels on the screen.
 *	Contains routines to intialize the screen, draw pixels
 *	to the screen, and close it when finished.
 *
 *		Macintosh Version
 *
 *	Tested with 16-bit (really 15) color on a Perform 630CD.
 *	It should work on a 24-bit machine, but has not been tested
 *	for that case.  It will need the SetScreenDepth call modified
 *	to work in that case.  On a machine with 8-bit maximum color
 *	it should be modified to use the grey-scale color map.  The
 *	author doesn't know how to do that.
 *	
 */

//Note: This should have a gamma correction routine in it too!

#include "aaline.h"

#include <time.h>
#include <math.h>
#include <Palettes.h>

#define rWindow 128


short GetScreenDepth(GDHandle screen);
void SetScreenDepth(GDHandle screen, short depth);

static Rect windRect;			// Rectangle for the window
static GDHandle gd;				// Graphical data?

static short old_depth;			// Original screen depth
static short depth;				// Frame buffer depth, in bits



/*
 * init_screen
 *
 *	Initialize the screen, open a 640 x 480 window, set it to
 *	the specified background color which is in bbggrr format.
 */

void
init_screen(int background)
{
	WindowPtr	mainPtr;
	OSErr		error;
	SysEnvRec	theWorld;

	RGBColor bg_color;						// The background color
	Rect bg_rect;							// The background rectangle

	// Make sure this machine can do color
	error = SysEnvirons(1, &theWorld);
	if (theWorld.hasColorQD == false) {
		// If no color, beep and exit
		SysBeep(50);
		close_screen();
	}

	// Initialize all the needed managers.
	InitGraf(&qd.thePort);
	InitWindows();

	// Get the window to draw in, set windRect to the interior size.
	mainPtr = GetNewCWindow(rWindow, nil, (WindowPtr) -1);
	windRect = mainPtr->portRect;

	SetPort(mainPtr);						// Set window to current graf port

	gd = GetMainDevice();
	old_depth = GetScreenDepth(gd);
	//fprintf(stderr, "Original screen depth: %d\n", depth);

	SetScreenDepth(gd, 16);
	depth = GetScreenDepth(gd);
	//fprintf(stderr, "New screen depth: %d\n", depth);

// Now clear the screen

	// Set the screen color to something reasonable
	bg_color.red = (background & 0xff) << 8;
	bg_color.green = ((background >> 8) & 0xff) << 8;
	bg_color.blue = ((background >> 16) & 0xff) << 8;
	RGBForeColor(&bg_color);

	// Set the rectangle to full screen and draw it
	SetRect(&bg_rect, 0, 0, WinWidth, WinHeight);
	PaintRect(&bg_rect);

} /* End of initialize_screen */



/*
 * close_screen
 *
 *	Wait for the user to click the mouse button, then put
 *	things back how they were and exit.
 */

void
close_screen(void)
{
	while (!Button())
		;

	/* Restore the screen to whatever it was when we started */
	SetScreenDepth(gd, old_depth);

	ExitToShell();
} /* End of close_screen */



/*
 * draw_pixel
 *
 *	Draw one pixel of the specified color at the specified location.
 *	Dither as best we can.
 */

void
draw_pixel(long int x, long int y, short r, short g, short b)
{
	RGBColor pixel_color;
	unsigned long dither_const;			/* Add to color, carry causes dither */
	unsigned long pixel_mask;			/* The bits to keep after dither */
	unsigned long r1, g1, b1;			/* New colors */

	/* Dither the least significant bit of representable range */

	pixel_mask = 0xffff0000u;			/* Which pixel bits to use */
	dither_const = 0x00000800u;			/* MSB - 4, for rounding */
	if (x & 0x02)
		dither_const |= 0x00001000u;	/* MSB - 3 */
	if ((x ^ y) & 0x02)
		dither_const |= 0x00002000u;	/* MSB - 2 */
	if (x & 0x01)
		dither_const |= 0x00004000u;	/* MSB - 1 */
	if ((x ^ y) & 0x01)
		dither_const |= 0x00008000u;	/* MSB */

	if (depth == 16) {
		/* 5 bits per component */
		dither_const >>= 5;
		pixel_mask >>= 5;
	}
	else if (depth == 24) {
		/* 8 bits per component */
		dither_const >>= 8;
		pixel_mask >>= 8;
	}
	else {
		/* Assume 4 bits per component */
		dither_const >>= 4;
		pixel_mask >>= 4;
	}

	/* Add dither adjustment */
	r1 = ((long) r & 0xffffu) + dither_const;
	g1 = ((long) g & 0xffffu) + dither_const;
	b1 = ((long) b & 0xffffu) + dither_const;

	/* Clamp any overflows */
	if ((r1 & 0xffff0000u) != 0)
		r1 = 0xffffu;
	if ((g1 & 0xffff0000u) != 0)
		g1 = 0xffffu;
	if ((b1 & 0xffff0000u) != 0)
		b1 = 0xffffu;

	/* Get rid of any extra bits */
	pixel_color.red = r1 & pixel_mask;
	pixel_color.green = g1 & pixel_mask;
	pixel_color.blue = b1 & pixel_mask;

	/* Finally, draw the pixel */
	RGBForeColor(&pixel_color);
	MoveTo(x, y);
	LineTo(x, y);

} /* End of write_pixel */



/*
 * GetScreenDepth -- Get Screen Depth
 */

short
GetScreenDepth(GDHandle screen)
{
    PixMapHandle    pixMap;         /* Pixel Map Handle */
    short           pixelSize;      /* Pixel Size */

    if (screen == nil) {
    	fprintf(stderr, "No screen\n");
        return (0);
    }

    pixMap = (*screen)->gdPMap;
    pixelSize = (*pixMap)->pixelSize;

    return (pixelSize);
} // End of GetScreenDepth



/*
 * SetScreenDepth -- Set Screen Depth
 */

void
SetScreenDepth(GDHandle screen, short depth)
{
    OSErr   error;                          /* Error */
    short   GetScreenDepth(GDHandle);       /* Get Screen Depth */
    void    PaintDesk(void);                /* Paint Desktop */

    if (screen == nil) {
        fprintf(stderr, "Screen is not defined\n");
        return;
    }

    if (GetScreenDepth(screen) == depth) {
        /* Screen is already at depth. */
        return;
    }

    if (HasDepth(screen, depth, 0, 0) == 0) {
        fprintf(stderr, "SetDepth Doesn't exist\n");
        return;
    }

    error = SetDepth(screen, depth, 0, 0);
    if (error != noErr) {
		fprintf(stderr, "SetDepth failure\n");
        return;
    }

    return;
} // End of SetScreenDepth

/* End of display_mac.c */
