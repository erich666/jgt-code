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

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author zwiers
 */
public class Vec3fTest {
    
    public Vec3fTest() {
    }

    @BeforeClass
    public static void setUpClass() throws Exception {
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
    }

    @Test
    public void equals() {       
        float[] a = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};    
        assertTrue(Vec3f.equals(a, 2, b, 3));
        assertTrue(! Vec3f.equals(a, 0, b, 0));
        
        float[] a3 = new float[] {1.0f, 2.0f, 3.0f};        
        
        float[] b3 = new float[] {1.0f, 2.0f, 3.0f};
        float[] c3 = new float[] {2.0f, 2.0f, 3.0f};
        float[] d3 = new float[] {1.0f, 4.0f, 3.0f};
        float[] e3 = new float[] {1.0f, 2.0f, 5.0f};
        assertTrue(Vec3f.equals(a3,b3));
        assertTrue(!Vec3f.equals(a3,c3));
        assertTrue(!Vec3f.equals(a3,d3));
        assertTrue(!Vec3f.equals(a3,e3));
        
        
    } /* Test of equals method, of class Vec3f. */

    @Test
    public void epsilonEquals() {
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};    
        assertTrue(Vec3f.epsilonEquals(a, 2, b, 3, 0.01f));
        assertTrue(! Vec3f.epsilonEquals(a, 0, b, 0, 0.01f));
        
        float[] a3 = new float[] {1.0f, 2.0f, 3.0f};        
        
        float[] b3 = new float[] {1.01f, 2.02f, 3.03f};
        float[] c3 = new float[] {2.0f, 2.0f, 3.0f};
        float[] d3 = new float[] {1.0f, 4.0f, 3.0f};
        float[] e3 = new float[] {1.0f, 2.0f, 5.0f};
        assertTrue(Vec3f.epsilonEquals(a3,b3, 0.05f));
        assertTrue(!Vec3f.epsilonEquals(a3,b3, 0.02f));
        assertTrue(!Vec3f.epsilonEquals(a3,c3, 0.1f));
        assertTrue(!Vec3f.epsilonEquals(a3,d3, 0.1f));
        assertTrue(!Vec3f.epsilonEquals(a3,e3, 0.1f));
        
    } /* Test of epsilonEquals method, of class Vec3f. */

    
    /**
     * Test of set method, of class Vec3f.
     * 1) set(float[] dst, int dstIndex, float[] src, int srcIndex)
     * 2) set(float[] dst, int dstIndex, float[] src, int srcIndex)
     * 3) set(float[] dst, int dstIndex, float x, float y, float z)
     * 4) set(float[] dst, float x, float y, float z)
     */
    @Test
    public void set() {
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect = new float[] {1.0f, 6.0f, 3.0f};
        Vec3f.set(a, 2, b, 1);
        assertTrue(Vec3f.equals(a, 2, expect, 0));
        
        float[] expect2 = new float[] {2.0f, 1.0f, 6.0f};
        Vec3f.set(a, b);
        assertTrue(Vec3f.equals(a, expect2));
        
        
        float[] a3 = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] expect3 = new float[] {2.0f, 1.0f, 6.0f};
        Vec3f.set(a3,2, 2.0f, 1.0f, 6.0f );
        assertTrue(Vec3f.equals(a3, 2, expect3, 0));
        
        float[] a4 = new float[3];        
        float[] expect4 = new float[] {2.0f, 1.0f, 6.0f};
        Vec3f.set(a4, 2.0f, 1.0f, 6.0f );
        assertTrue(Vec3f.equals(a4, expect4));
    }

    
 
    
    /**
     * Test of add method, of class Vec3f.
     * 1) add(float[] dst, int dstIndex, float[] a, int aIndex, float[] b, int bIndex)
     * 2) add(float[] dst, int dstIndex, float[] a, int aIndex)
     * 3) add(float[] dst, float[] a, float[] b
     * 4) add(float[] dst, float[] a)
     */
    @Test
    public void add() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect = new float[] {4.003f, 10.004f, 8.005f};
        Vec3f.add(dest, 4, a, 2, b, 1);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-6f));
        
        float[] a2 = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b2 = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect2 = new float[] {4.003f, 10.004f, 8.005f};
        Vec3f.add(a2, 2, b2, 1);
        assertTrue(Vec3f.epsilonEquals(a2, 2, expect2, 0, 1E-6f));
        
        float[] dest3 = new float[3]; 
        float[] a3 = new float[] {3.003f, 4.004f, 5.005f};        
        float[] b3 = new float[] {1.0f, 6.0f, 3.0f};
        float[] expect3 = new float[] {4.003f, 10.004f, 8.005f};
        Vec3f.add(dest3, a3, b3);
        assertTrue(Vec3f.epsilonEquals(dest3,expect3, 1E-6f));
        
        float[] a4 = new float[] {3.003f, 4.004f, 5.005f};        
        float[] b4 = new float[] {1.0f, 6.0f, 3.0f};
        float[] expect4 = new float[] {4.003f, 10.004f, 8.005f};
        Vec3f.add(a4, b4);
        //System.out.println("a="+Vec3f.toString(a, 2));
        assertTrue(Vec3f.epsilonEquals(a4,expect4, 1E-6f));
    }
    
    
    /**
     * Test of add method, of class Vec3f.
     * 1) sub(float[] dst, int dstIndex, float[] a, int aIndex, float[] b, int bIndex)
     * 2) sub(float[] dst, int dstIndex, float[] a, int aIndex)
     * 3) sub(float[] dst, float[] a, float[] b)
     * 4) sub(float[] dst, float[] a)
     */
    @Test
    public void sub() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect = new float[] {2.003f, -1.996f, 2.005f};
        Vec3f.sub(dest, 4, a, 2, b, 1);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-6f));
        
        float[] a2 = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b2 = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect2 = new float[] {2.003f, -1.996f, 2.005f};
        Vec3f.sub(a2, 2, b2, 1);
        assertTrue(Vec3f.epsilonEquals(a2, 2, expect2, 0, 1E-6f));
        
        float[] dest3 = new float[3]; 
        float[] a3 = new float[] {3.003f, 4.004f, 5.005f};        
        float[] b3 = new float[] {1.0f, 6.0f, 3.0f};
        float[] expect3 = new float[] {2.003f, -1.996f, 2.005f};
        Vec3f.sub(dest3, a3, b3);
        assertTrue(Vec3f.epsilonEquals(dest3,expect3, 1E-6f));
        
        float[] a4 = new float[] {3.003f, 4.004f, 5.005f};        
        float[] b4 = new float[] {1.0f, 6.0f, 3.0f};
        float[] expect4 = new float[] {2.003f, -1.996f, 2.005f};
        Vec3f.sub(a4, b4);
        assertTrue(Vec3f.epsilonEquals(a4, expect4, 1E-6f));
    }
    
   
    /**
     * Test of scale method, of class Vec3f.
     * 1) scale(float scale, float[] dst, int dstIndex)
     * 2) scale(float scale, float[] dst)
     */
    @Test
    public void scale() {
        float[] a1 = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.00f, 6.00f};
        float[] a2 = new float[] {3.003f, 4.004f, 5.00f};
        float[] expect = new float[] {9.009f, 12.012f, 15.00f};
        Vec3f.scale(3.0f, a1, 2);
        assertTrue(Vec3f.epsilonEquals(a1, 2, expect, 0, 1E-6f));
        Vec3f.scale(3.0f, a2);
        assertTrue(Vec3f.epsilonEquals(a2, expect, 1E-6f));
    }

    /**
     * Test of scaleAdd method, of class Vec3f.
     * 1) scaleAdd(float[] dst, int dstIndex, float scale, float[] a, int aIndex, float[] b, int bIndex)
     * 2) scaleAdd(float scale, float[] dst, int dstIndex, float[] a, int aIndex)
     * 3) scaleAdd(float[] dst, float scale, float[] a, float[] b)
     * 4) scaleAdd(float scale, float[] dst, float[] a)
     */
    @Test
    public void scaleAdd() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect = new float[] {10.009f, 18.012f, 18.015f};
        // 1)
        Vec3f.scaleAdd(dest, 4, 3.0f,  a, 2, b, 1);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-5f));
        
        //2)
        Vec3f.scaleAdd(3.0f,  a, 2, b, 1);
        assertTrue(Vec3f.epsilonEquals(a, 2, expect, 0, 1E-5f));
        
        float[] dest1 = new float[3]; 
        float[] a1 = new float[] { 3.003f, 4.004f, 5.005f};        
        float[] b1 = new float[] {1.0f, 6.0f, 3.0f};
        //3)
        Vec3f.scaleAdd(dest1, 3.0f,  a1,  b1);
        assertTrue(Vec3f.epsilonEquals(dest1, expect, 1E-5f));
        //4) 
        Vec3f.scaleAdd(3.0f,  a1,  b1);
        assertTrue(Vec3f.epsilonEquals(a1, expect, 1E-5f));
        
    }

    /**
     * Test of negate method, of class Vec3f.
     * 1) negate(float[] dst, int dstIndex, float[] src, int srcIndex
     * 2) negate(float[] dst, int dstIndex) 
     * 3) negate(float[] dst, float[]src)
     * 4) negate(float[] dst)
     */
    @Test
    public void negate() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.002f, 3.003f, 4.004f, 5.005f, 6.006f};        
        float[] expect = new float[] {-3.003f, -4.004f, -5.005f};
        // 1)
        Vec3f.negate(dest, 4, a, 2);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-6f));
        // 2)
        Vec3f.negate(a, 2);
        assertTrue(Vec3f.epsilonEquals(a, 2, expect, 0, 1E-6f));
        
        float[] dest1 = new float[3]; 
        float[] a1 = new float[] { 3.003f, 4.004f, 5.005f};    
        
        // 3)
        Vec3f.negate(dest1, a1);
        assertTrue(Vec3f.epsilonEquals(dest1, expect, 1E-6f));
        // 4)
        Vec3f.negate(a1);
        assertTrue(Vec3f.epsilonEquals(a1, expect, 1E-6f));
    }

    /**
     * Test of cross method, of class Vec3f.
     * 1) cross(float[] dst, int dstIndex, float[] a, int aIndex, float[] b, int bIndex) 
     * 2) cross(float[] dst, float[] a, float[] b)
     */
    @Test
    public void cross() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.002f, 3.0f, 4.0f, 5.0f, 6.006f};        
        float[] b = new float[] {2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 5.0f};
        float[] expect = new float[] {-18.0f, -4.0f, 14.0f};
        // 1)
        Vec3f.cross(dest, 4, a, 2, b, 1);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-5f));
        
        //2)
        float[] dest2 = new float[3]; 
        float[] a2 = new float[] {3.0f, 4.0f, 5.0f};        
        float[] b2 = new float[] {1.0f, 6.0f, 3.0f};
        Vec3f.cross(dest2, a2, b2);
        assertTrue(Vec3f.epsilonEquals(dest2,expect, 1E-5f));
    }

    /**
     * Test of dot method, of class Vec3f.
     * 1) dot(float[] a, int aIndex, float[] b, int bIndex)
     * 2) dot(float[] a, float[] b
     */
    @Test
    public void dot() {
        float result = 0.0f;
        float[] a = new float[] {1.001f, 2.002f, 3.0f, 4.0f, 5.0f, 6.006f};        
        float[] b = new float[] {2.0f, 2.0f, 6.0f, 4.0f, 4.0f, 5.0f};
        float expect = 50.0f;
        // 1)
        result = Vec3f.dot(a, 2, b, 1);
        assertTrue(Math.abs(result-expect) < 1E-6f);
        
        //2)
        float[] a2 = new float[] {3.0f, 4.0f, 5.0f};        
        float[] b2 = new float[] {2.0f, 6.0f, 4.0f};
        float result2 = Vec3f.dot(a2, b2);
        assertTrue(Math.abs(result2-expect) < 1E-6f);
    }

    /**
     * Test of lengthSq method, of class Vec3f.
     * 1) lengthSq(float[] a, int aIndex)
     * 2) lengthSq(float[] a)
     */
    @Test
    public void lengthSq() {
        float result = 0.0f;
        float[] a = new float[] {1.001f, 2.002f, 2.0f, 3.0f, 4.0f, 6.006f};        
        float expect = 29.0f;
        // 1)
        result = Vec3f.lengthSq(a, 2);
        assertTrue(Math.abs(result-expect) < 1E-6f);
        
        //2)
        float[] a2 = new float[] {2.0f, 3.0f, 4.0f};        
        float result2 = Vec3f.lengthSq(a2);
        assertTrue(Math.abs(result2-expect) < 1E-6f);
    }

    /**
     * Test of length method, of class Vec3f.
     * 1) length(float[] a, int aIndex)
     * 2) length(float[] a)
     */
    @Test
    public void length() {
        float result = 0.0f;
        float[] a = new float[] {1.001f, 2.002f, 2.0f, 3.0f, 4.0f, 6.006f};        
        float expect = 29.0f;
        // 1)
        result = Vec3f.length(a, 2);
        assertTrue(Math.abs(result*result-expect) < 1E-6f);
        
        //2)
        float[] a2 = new float[] {2.0f, 3.0f, 4.0f};        
        float result2 = Vec3f.length(a2);
        assertTrue(Math.abs(result2*result2-expect) < 1E-6f);
    }

    /**
     * Test of interpolate method, of class Vec3f.
     * 1) interpolate(float[] dst, int dstIndex, float[] a, int aIndex, float[] b, int bIndex, float alpha )
     * 2) interpolate(float[] dst, float[] a, float[] b, float alpha )
     */
    @Test
    public void interpolate() 
    {
       
    }

    /**
     * Test of normalize method, of class Vec3f.
     * 1) normalize(float[] dst, int dstIndex, float[] a, int aIndex)
     * 2) normalize(float[] a, int aIndex
     * 3) normalize(float[] dst, float[] a)
     * 4) normalize(float[] a)
     */
    @Test
    public void normalize() {
        float[] dest = new float[10]; 
        float[] a = new float[] {1.001f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};     
        float len = (float) Math.sqrt(50.0);
        float[] expect = new float[] {3.0f/len, 4.0f/len, 5.0f/len};
        // 1)
        Vec3f.normalize(dest, 4, a, 2);
        float newLen = Vec3f.length(dest, 4);
        assertTrue(Vec3f.epsilonEquals(dest, 4, expect, 0, 1E-5f));
        assertTrue(Math.abs(newLen-1.0f) < 1E-5f);
        
        //2)
        Vec3f.normalize(a, 2);
        float newLen2 = Vec3f.length(dest, 4);
        assertTrue(Vec3f.epsilonEquals(a, 2, expect, 0, 1E-5f));
        assertTrue(Math.abs(newLen2-1.0f) < 1E-5f);
        
        float[] dest2 = new float[3]; 
        float[] a2 = new float[] { 3.0f, 4.0f, 5.0f};        
        //3)
        Vec3f.normalize(dest2, a2);
        assertTrue(Vec3f.epsilonEquals(dest2, expect, 1E-5f));
        //4) 
        Vec3f.normalize(a2);
        assertTrue(Vec3f.epsilonEquals(a2, expect, 1E-5f));
        
    }

    /**
     * Test of toString method, of class Vec3f.
     * 1) toString(float[] a, int index)
     * 2) toString(float[] a)
     */
    @Test
    public void toStringTest() {
        float[] a = new float[] {1.001f, 2.0f, 3.1f, 4.2f, 5.3f, 6.0f};    
        float[] a2 = new float[] {3.1f, 4.2f, 5.3f};   
        String expect = "(3.1, 4.2, 5.3)";
        String result = Vec3f.toString(a, 2);
        String result2 = Vec3f.toString(a2);
        assertTrue(result.equals(expect));
        assertTrue(result2.equals(expect));
  
    }
  
}
