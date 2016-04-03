// created by Shaun D. Ramsey and Kristin Potter (c) 2003
// email ramsey()cs.utah.edu with questions/comments

/*
The ray bilinear patch intersection software are "Open Source"  
according to the MIT License located at:
 http://www.opensource.org/licenses/mit-license.php

Copyright (c) 2003 Shaun David Ramsey, Kristin Potter, Charles Hansen

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sel copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
*/

#include "bilinear.h"
#include <iostream.h>


int main()
{
  //create 4 points
  Vector P00(0, 0,0);
  Vector P01(3, 1,3);
  Vector P10(1, 3 ,1);
  Vector P11(1, -2,4);
  // make them into a bilinear patch
  BilinearPatch bp(P00,P01,P10,P11);
  //you have some ray information
  Vector r(1,0.3,10); //origin of the ray
  Vector q(.100499,0,-.994937); // a ray direction
  q.normalize();
  Vector uv; // variables returned
  Vector normal;
  if(bp.RayPatchIntersection(r,q,uv)) // run intersection test
    {
      cout << "Intersected the patch at point " <<
	bp.SrfEval(uv.x(),uv.y()) << " with t=" << uv.z() << endl;
      normal = 	bp.Normal(uv.x(),uv.y());
      cout << "The normal at this point is " <<  normal << endl;
      cout << "normalized " << normal.normal() << endl;

    }
  else
    cout << "Did not intersect the patch:" << uv <<endl;
  

}