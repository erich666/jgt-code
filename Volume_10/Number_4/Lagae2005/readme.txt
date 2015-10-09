Sample code for the article

Ares Lagae and Philip Dutré.
An efficient ray-quadrilateral intersection test.
Journal of graphics tools, 10(4):23-32, 2005

If you use this code please cite the article.

Copyright (c) 2004 Ares Lagae

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


This program is a minimal ray tracer. It ray traces a single quad, covered
with a checkerboard texture, and outputs the image as "image.ppm".

The classes provided in this file are not complete, only operations needed
to implement the intersection algorithm and ray tracer are provided.

To compile and run:

  $ g++ erqit.cpp -o erqit
  $ ./erqit

See https://github.com/erich666/jgt-code/tree/master/Volume_10/Number_4/Lagae2005
for the most recent version of this file and additional documentation.

Revision history
 2004-10-08  initial version
 2004-03-11  minor changes
 2006-04-14  minor changes
 2015-10-08  minor update for new JGT repository
