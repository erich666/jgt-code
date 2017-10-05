/*
 * display_unix.c
 *
 *	Machine specific code to display pixels on the screen.
 *	Contains routines to intialize the screen, draw pixels
 *	to the screen, and close it when finished.
 *
 *		Unix X11 Version
 *
 *	Tested on a Sun Ultra1 Creator3D.  This may need a little
 *	tweaking to get proper behavior on other machines.
 */

/* Note: This should have a gamma correction routine in it too! */

#include "aaline.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define NUMCELLS 225

Display *dpy;
Window win;
Window	frame;
Colormap cmap;
GC mygc;
int scr;
Visual *vis;

int pmap[NUMCELLS];

static int visClasses[] = {
   TrueColor,
   DirectColor,
   PseudoColor,
   StaticColor,
   GrayScale,
   StaticGray,
};



/*
 * init_screen
 *
 *	Initialize the screen, open a 640 x 480 window, set it to
 *	the specified background color which is in bbggrr format.
 */

void
init_screen(int background)
{
   int done, i, j;
   XVisualInfo	template, *vislist, *ptr;
   XSetWindowAttributes	attributes;
   Visual *defvis;
   int black, white;
   int planes, nvis;

   dpy = XOpenDisplay("");
   if (0 == dpy) {
      printf("%s\n", "Unable to open display");
      exit (1);
   }

   scr = DefaultScreen(dpy);
   vis = defvis = DefaultVisual(dpy, scr);
   planes = DisplayPlanes(dpy, scr);
   printf("default planes = %d\n", planes);

   ptr = NULL;
   bzero(&template, sizeof(template)) ;
   for(i=0; i<sizeof(visClasses)/sizeof(visClasses[0]); i++) {

      template.class = visClasses[i] ;
      vislist = XGetVisualInfo(dpy, VisualClassMask, &template, &nvis);

      for (j=0; j<nvis; j++)
	 if ((vislist[j].screen == scr) &&
	      (vislist[j].depth > planes)) {
	    planes = vislist[j].depth;
	    vis = vislist[j].visual;
	    break;
	 }
   }

   printf("planes used = %d\n", planes);

   black = BlackPixel(dpy, scr);
   white = WhitePixel(dpy, scr);
   frame = XCreateSimpleWindow(dpy, RootWindow(dpy, scr),
			       0, 0, WinWidth, WinHeight, 0, black, black );

   if (vis == defvis)  {
      int i;
      XColor c;
      printf("using default visual\n");
      cmap = DefaultColormap(dpy, scr);
      for (i=0; i<NUMCELLS; i++) {
	 int val = i* (float)(65536.0/NUMCELLS);
	 c.red = val;
	 c.green = val;
	 c.blue = val;
	 XAllocColor(dpy, cmap, &c);
	 pmap[i] = c.pixel;
      }
   }
   else {
      /* printf("NOT using default visual\n"); */
      cmap = XCreateColormap(dpy, frame, vis, AllocNone);
   }

   bzero(&attributes, sizeof(XSetWindowAttributes)) ;
   attributes.border_pixel = black;
   attributes.background_pixel = black;
   attributes.colormap = cmap;
   win = XCreateWindow(dpy, frame, 0, 0, WinWidth, WinHeight, 0,
		       planes, InputOutput, vis,
		       CWBackPixel|CWBorderPixel|CWColormap, &attributes);

   mygc = XCreateGC(dpy, win, 0, 0);
   XSetBackground(dpy, mygc, black);

   XSelectInput(dpy, win, ExposureMask);

   XMapWindow(dpy, frame);
   XMapSubwindows(dpy, frame);

#if 0
   XSetForeground(dpy, mygc, 0xff);
   XFillRectangle(dpy, win, mygc, 0, 0, WinWidth, WinHeight);
   XDrawPoint(dpy, win, mygc, 5, 5);
   XSync(dpy);
   XFlush(dpy);
#endif

    done = 0;
    while (done == 0) {
	XEvent myevent;

	XNextEvent(dpy, &myevent);

	switch (myevent.type) {
	  case Expose:
	    if (myevent.xexpose.count == 0) {
		done = 1;
		if ((vis->class == TrueColor) || (vis->class == DirectColor)) {
		    XSetForeground(dpy, mygc, background);
		}
		else {
		    XColor bg;

		    bg.red = (background & 0xff) << 8;
		    bg.green = ((background >> 8) & 0xff) << 8;
		    bg.blue = ((background >> 16) & 0xff) << 8;
		    XAllocColor(dpy, cmap, &bg);
		    XSetForeground(dpy, mygc, bg.pixel);
		}
		XFillRectangle(dpy, win, mygc, 0, 0, WinWidth, WinHeight);
	    }
	}
    }

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
    int done;

    XFlush(dpy);

    XSelectInput(dpy, win, ButtonPressMask | KeyPressMask);

    done = 0;
    while (done == 0) {
	XEvent myevent;

	XNextEvent(dpy, &myevent);
	switch(myevent.type) {
	  case ButtonPress:
	  case KeyPress:
	    done = 1;
	    break;
	}
    }

    XFreeGC(dpy, mygc);
    XFreeColormap(dpy, cmap);
    XDestroyWindow(dpy, win);
    XDestroyWindow(dpy, frame);
    XCloseDisplay(dpy);
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
    float grey;				/* Grey-scale equivalent value */
    int index;				/* Color index value */
    unsigned short rr, gg, bb;		/* Color values */

    if ((vis->class == TrueColor) || (vis->class == DirectColor))
	XSetForeground(dpy, mygc,
		       ((b >> 8) & 0xff) << 16 |
		       ((g >> 8) & 0xff) << 8 | ((r >> 8) & 0xff));
    else {
	rr = r & 0xffff;
	gg = g & 0xffff;
	bb = b & 0xffff;

	grey = 0.299 * (float)rr + 0.587 * (float)gg + 0.114 * (float)bb;
	index = (int)((grey * NUMCELLS) / 65535.0);
	XSetForeground(dpy, mygc, pmap[index]);
    }

    XDrawPoint(dpy, win, mygc, x, y);

} /* End of write_pixel */

/* End of display_unix.c */
