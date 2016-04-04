/*
 * version.c
 *
 * Copyright (C) 1995 Tien-tsin Wong
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
 */
#include "libcommon/common.h"
#include "stats.h"

void
VersionPrint()
{
  fprintf(Stats.fstats,	"sampexp: ver. 1.0\n");
}
