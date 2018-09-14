/*
Copyright (c) 2008 Human Media Interaction, University of Twente
Web: http://hmi.ewi.utwente.nl/

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/
package hmi.math;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SpatialVecTest
{

    @Before
    public void setUp() throws Exception
    {
    }

    @After
    public void tearDown() throws Exception
    {
    }

    @Test
    public void testSet()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void testAddFloatArrayFloatArrayFloatArray()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void testAddFloatArrayFloatArray()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void testDot()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void testCross()
    {
        float a[]=new float[6];
        float b[]=new float[6];
        float dst[] = new float[6];
        for(int i=0;i<6;i++)
        {
            a[i] = i+1;
            b[i] = 3-i;
        }
        //axa=0
        SpatialVec.cross(dst, a, a);
        //System.out.println(SpatialVec.toString(dst));
        assertTrue(SpatialVec.epsilonEquals(dst, SpatialVec.ZERO, 0.0001f));
        
        //dst = axb => a.dst=b.dst=0
        SpatialVec.cross(dst, a, b);
        assertTrue(Math.abs(SpatialVec.dot(a,dst))<=0.00001);
        assertTrue(Math.abs(SpatialVec.dot(b,dst))<=0.00001);
        
        //Same test on indexed version
        a = new float[12];
        b = new float[12];
        dst = new float[12];
        for(int i=0;i<6;i++)
        {
            a[i] = 0;
            b[i] = 0;
        }
        for(int i=6;i<12;i++)
        {
            a[i] = i+1;
            b[i] = 3-i;
        }
        //axa=0
        SpatialVec.cross(dst,6, a,6, a,6);
        assertTrue(SpatialVec.epsilonEquals(dst, 6, SpatialVec.ZERO, 0, 0.0001f));
        
        //dst = axb => a.dst=b.dst=0
        SpatialVec.cross(dst,6, a,6, b,6);
        assertTrue(Math.abs(SpatialVec.dot(a,6,dst,6))<=0.00001);
        assertTrue(Math.abs(SpatialVec.dot(b,6,dst,6))<=0.00001); 
    }
    
    
    @Test
    public void testCrossForce()
    {
        float a[]=new float[6];
        float b[]=new float[6];
        float dst[] = new float[6];
        for(int i=0;i<6;i++)
        {
            a[i] = i+1;
            b[i] = 3-i;
        }
        //axa=0
        SpatialVec.crossForce(dst, a, a);
        //dst = axb => a.dst=b.dst=0
        SpatialVec.crossForce(dst, a, b);
        assertTrue(Math.abs(SpatialVec.dot(a,dst))<=0.00001);
        assertTrue(Math.abs(SpatialVec.dot(b,dst))<=0.00001);
        
        //Same test on indexed version
        a = new float[12];
        b = new float[12];
        dst = new float[12];
        for(int i=0;i<6;i++)
        {
            a[i] = 0;
            b[i] = 0;
        }
        for(int i=6;i<12;i++)
        {
            a[i] = i+1;
            b[i] = 3-i;
        }
        //axa=0
        SpatialVec.crossForce(dst,6, a,6, a,6);
        
        //dst = axb => a.dst=b.dst=0
        SpatialVec.crossForce(dst,6, a,6, b,6);
        assertTrue(Math.abs(SpatialVec.dot(a,6,dst,6))<=0.00001);
        assertTrue(Math.abs(SpatialVec.dot(b,6,dst,6))<=0.00001); 
    }
}
