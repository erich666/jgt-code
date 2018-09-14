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

public class SpatialTransformTest
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
    public void testTransform()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void testMul()
    {
        //fail("Not yet implemented");
    }

    @Test
    public void transformTranspose()
    {
        float trans[]=new float[12];
        float transT[]=new float[12];
        float dsta[]=new float[6];
        float dstb[]=new float[6];
        float r[] = new float[3];
        float m[] = new float[9];
        float q[] = new float[4];
        float v[] = new float[6];
        for(int i=0;i<6;i++)
        {
            v[i]=i+1;
        }
        
        for(int i=0;i<3;i++)
            r[i]=i+1;
        Quat4f.setFromAxisAngle4f(q, 1f, 0.8f, -0.3f, 1.1f);
        Mat3f.setFromQuatScale(m, q, 1);
        SpatialTransform.setFromMat3fVec3f(trans, m, r);
        SpatialTransform.transpose(transT,trans);
        
        SpatialTransform.transformMotion(dsta, transT, v);
        SpatialTransform.transformMotionTranspose(dstb, trans, v);
        
        assertTrue(SpatialVec.epsilonEquals(dsta, dstb, 0.0005f));
    }
    
    @Test
    public void transformForceTranspose()
    {
        float trans[]=new float[12];
        float transT[]=new float[12];
        float dsta[]=new float[6];
        float dstb[]=new float[6];
        float r[] = new float[3];
        float m[] = new float[9];
        float q[] = new float[4];
        float v[] = new float[6];
        for(int i=0;i<6;i++)
        {
            v[i]=i+1;
        }
        
        for(int i=0;i<3;i++)
            r[i]=i+1;
        Quat4f.setFromAxisAngle4f(q, 1f, 0.8f, -0.3f, 1.1f);
        Mat3f.setFromQuatScale(m, q, 1);
        SpatialTransform.setFromMat3fVec3f(trans, m, r);
        SpatialTransform.transpose(transT,trans);
        
        SpatialTransform.transformMotion(dsta, transT, v);
        SpatialTransform.transformMotionTranspose(dstb, trans, v);
        
        assertTrue(SpatialVec.epsilonEquals(dsta, dstb, 0.0005f));
    }
    
    @Test
    public void testTranspose()
    {
        float trans[]=new float[12];
        float transT[]=new float[12];
        float dst[]=new float[12];
        float r[] = new float[3];
        float m[] = new float[9];
        float q[] = new float[4];
        for(int i=0;i<3;i++)
            r[i]=i+1;
        Quat4f.setFromAxisAngle4f(q, 1f, 0.8f, -0.3f, 1.1f);
        Mat3f.setFromQuatScale(m, q, 1);
        SpatialTransform.setFromMat3fVec3f(trans, m, r);
        SpatialTransform.transpose(transT,trans);
        SpatialTransform.mul(dst, trans, transT);
        
        //transT*trans=I
        assertTrue(SpatialTransform.epsilonEquals(dst, SpatialTransform.ID, 0.0005f));
        SpatialTransform.transpose(transT);
        
        //(transT)^T = trans, tests transpose in place
        assertTrue(SpatialTransform.epsilonEquals(trans, transT, 0.0005f));
    }

}
