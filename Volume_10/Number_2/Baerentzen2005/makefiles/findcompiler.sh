#!/bin/sh
if (CC -V  > /dev/null 2>&1)
    then COMPILER="CC"
elif (CC -version > /dev/null 2>&1)
    then COMPILER="CC"
elif (g++ -v > /dev/null 2>&1)
    then COMPILER="g++";
elif (icc -help > /dev/null 2>&1)
    then COMPILER="icc"
elif (g++3 -v > /dev/null 2>&1)
    then COMPILER="g++3"
else COMPILER="NO_COMPILER_FOUND"
fi
echo $COMPILER
