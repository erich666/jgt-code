/*
 * main.c
 *
 * Copyright (C) 1989, 1991, Craig E. Kolb
 * All rights reserved.
 *
 * This software may be freely copied, modified, and redistributed
 * provided that this copyright notice is preserved on all copies.
 *
 * You may not distribute this software, in whole or in part, as part of
 * any commercial product without the express consent of the authors.
 *
 * There is no warranty or other guarantee of fitness of this software
 * for any purpose.  It is provided solely "as is".
 *
 * $Id: main.c,v 4.0 91/07/17 14:50:39 kolb Exp Locker: kolb $
 *
 * $Log:	main.c,v $
 * Revision 4.0  91/07/17  14:50:39  kolb
 * Initial version.
 * 
 */

char rcsid[] = "$Id: main.c,v 4.0 91/07/17 14:50:39 kolb Exp Locker: kolb $";

#include "rayshade.h"
#include "options.h"
#include "stats.h"
#include "viewing.h"
#include "picture.h"

int
#ifdef LINDA
rayshade_main(argc, argv)
#else
main(argc, argv)
#endif
int argc;
char **argv;
{
	Float utime, stime, lasttime;
	int i;
	extern Geom *World;

#ifdef LINDA
	Options.workernum = 0;	/* we're the supervisor */
#endif

	RSInitialize(argc, argv);


	/*
	 * Start the first frame.
	 */
	RSStartFrame(Options.startframe);
	/*
 	 * Print more information than we'll ever need to know...
	 */
	if (Options.verbose) {
		/* World object info. */
		AggregatePrintInfo(World, Stats.fstats);
		/* Print info about rendering options and the like. */
		RSOptionsList();
	}
	/*
	 * Start new picture.
	 */
	PictureStart(argv);
	/*
	 * Print preprocessing time.
	 */
	RSGetCpuTime(&utime, &stime);
	fprintf(Stats.fstats,"Preprocessing time:\t");
	fprintf(Stats.fstats,"%2.2fu  %2.2fs\n", utime, stime);
	fprintf(Stats.fstats,"Starting trace.\n");
	(void)fflush(Stats.fstats);
	lasttime = utime+stime;
	/*
	 * Render the first frame
	 */
	raytrace(argc, argv);
	/*
	 * Render the remaining frames.
	 */
	for (i = Options.startframe +1; i <= Options.endframe ; i++) {
		PictureFrameEnd();	/* End the previous frame */
		RSGetCpuTime(&utime, &stime);
		fprintf(Stats.fstats, "Total CPU time for frame %d: %2.2f \n", 
			i - 1, utime+stime - lasttime);
		PrintMemoryStats(Stats.fstats);
		(void)fflush(Stats.fstats);
		lasttime = utime+stime;
		RSStartFrame(i);
		if (Options.verbose) {
			AggregatePrintInfo(World, Stats.fstats);
			(void)fflush(Stats.fstats);
		}
		PictureStart(argv);
		raytrace(argc, argv);
	}
	/*
	 * Close the image file.
	 */
	PictureFrameEnd();	/* End the last frame */
	PictureEnd();
	StatsPrint();
	return 0;
}
