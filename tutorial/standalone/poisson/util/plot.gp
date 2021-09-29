#!/usr/bin/env gnuplot

# Get extenst from environment
XLEN="`echo $XLEN`"
YLEN="`echo $YLEN`"

set dgrid3d XLEN, YLEN
set terminal x11
splot "`echo $DATFILE`" u 1:2:3 w pm3d palette
pause -1
