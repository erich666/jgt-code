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
import org.junit.*;

/**
 * JUnit test for hmi.math.Mat4f 
 */
public class Mat4fTest {

   float[] m, m1, m2, mexpect;

   @Before
   public  void setUp()  { // common initialization, for all tests
      m = new float[16];
      m1 = new float[16];
      m2 = new float[16];
      mexpect = new float[16];
   }

   public void initZero(float[] m) {
       Mat4f.set(m, 0.0f,  0.0f,  0.0f,  0.0f,
                    0.0f,  0.0f,  0.0f,  0.0f,
                    0.0f,  0.0f,  0.0f,  0.0f,
                    0.0f,  0.0f,  0.0f,  0.0f);
   }

   public void initId(float[] m) {
       Mat4f.set(m, 1.0f,  0.0f,  0.0f,  0.0f,
                    0.0f,  1.0f,  0.0f,  0.0f,
                    0.0f,  0.0f,  1.0f,  0.0f,
                    0.0f,  0.0f,  0.0f,  1.0f);
   }

   public void init1(float[] m) {
       Mat4f.set(m, 1.0f,  2.0f,  3.0f,  4.0f,
                    5.0f,  6.0f,  7.0f,  8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                   13.0f, 14.0f, 15.0f, 16.0f);
   }

   public void showM() {
      System.out.println("m=\n" + Mat4f.toString(m));
   }

   public void showMexpect() {
      System.out.println("mexpect=\n" + Mat4f.toString(mexpect));
   }

   @After
   public  void tearDown() { 
   }

   @Test
   public void testScale() {
      init1(m);
      Mat4f.scale(m, 2.0f);
      Mat4f.set(mexpect, 2.0f,  4.0f,  6.0f,  4.0f,
                        10.0f, 12.0f, 14.0f,  8.0f,
                        18.0f, 20.0f, 22.0f, 12.0f,
                        13.0f, 14.0f, 15.0f, 16.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));
   }

   
   @Test
    public void testFrom3x3() {
       float[] m3x3 = new float[9];
       Mat3f.set(m3x3, 
             1.0f,  2.0f,  3.0f, 
             5.0f,  6.0f,  7.0f, 
             9.0f, 10.0f, 11.0f
       ); 
       float[] mexpect = new float[16];
       Mat4f.set(mexpect, 
              1.0f,  2.0f,  3.0f,  0.0f,
              5.0f,  6.0f,  7.0f,  0.0f,
              9.0f, 10.0f, 11.0f,  0.0f,
              0.0f, 0.0f,   0.0f,  1.0f
       ); 
       float[] m4x4 = Mat4f.from3x3(m3x3);
       assertTrue(Mat4f.equals(m4x4, mexpect));
       
    }
   
   
   @Test
   public void testGetScalingMatrix() {
      float[] scale = new float[]{2.0f, 3.0f, 4.0f};
      m = Mat4f.getScalingMatrix(scale);
      Mat4f.set(mexpect, 2.0f,  0.0f,  0.0f,  0.0f,
                         0.0f,  3.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  4.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));
   }

   @Test
   public void testGetTranslationMatrix() {
      float[] trans = new float[]{2.0f, 3.0f, 4.0f};
      m = Mat4f.getTranslationMatrix(trans);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  2.0f,
                         0.0f,  1.0f,  0.0f,  3.0f,
                         0.0f,  0.0f,  1.0f,  4.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));     
   }



   @Test
   public void testSetFloatArrayFloatArray() {
      initZero(m1);
      init1(m2);
      init1(mexpect);
      Mat4f.set(m1, m2);
      assertTrue(Mat4f.epsilonEquals(m1, mexpect, 0.0001f));  // m1 is set correctly?
      assertTrue(Mat4f.epsilonEquals(m2, mexpect, 0.0001f));  // m2 not modified?      
   }

   @Test(expected= IllegalArgumentException.class)
   public void testGetSkewMatrixErrorArguments1()  {
      
      float angle = 90.0f;  // illegal angle, so expect an IllegalArgumentException
      float[] rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      float[] tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(false); // should arrive here
   }

   @Test(expected= IllegalArgumentException.class)
   public void testGetSkewMatrixErrorArguments2()  {
      
      float angle = -90.0f;  // illegal angle, so expect an IllegalArgumentException
      float[] rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      float[] tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(false); // should arrive here
   }

   @Test
   public void testGetSkewMatrixBasics() {
      
      float angle = 0.0f;  // Zero angle, so expect the identity matrix
      float[] rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      float[] tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(null, angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
      
      angle = 45.0f;
      rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
      
      angle = 60.0f;                          // non-45 degrees angle
      rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      float rt3 = (float) Math.sqrt(3.0); // = +- 1.73
      Mat4f.set(mexpect, 1.0f,  rt3,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
      
      angle = 89.0f;                          // large, but still legal angle
      rvec = new float[]{ 0.0f, 1.0f, 0.0f }; // Y-axis
      tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
     
      Mat4f.set(mexpect, 1.0f,  57.289f,  0.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.01f)); 
      
      
      angle = 45.0f;
      rvec = new float[]{ 0.0f, 0.0f, 1.0f };  // Z-axis
      tvec = new float[]{ 1.0f, 0.0f, 0.0f };  // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));    
       
      angle = 45.0f;
      rvec = new float[]{ 1.0f, 0.0f, 0.0f };  // X-axis
      tvec = new float[]{ 0.0f, 1.0f, 0.0f };  // Y-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,
                         1.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));     
       
      angle = -45.0f;  // negative angle
      rvec = new float[]{ 1.0f, 0.0f, 0.0f };  // X-axis
      tvec = new float[]{ 0.0f, 1.0f, 0.0f };  // Y-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,
                         -1.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));         
   }

   @Test
   public void testGetSkewMatrixNonOrthogonal() {
      float angle = 15.0f;
      float[] rvec = new float[]{ 1.0f, 1.0f, 0.0f }; // 45 degree rvec
      float[] tvec = new float[]{ 1.0f, 0.0f, 0.0f }; // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      float rt3 = (float) Math.sqrt(3.0); // = +- 1.73
      Mat4f.set(mexpect, 1.0f,  rt3-1.0f,  0.0f,  0.0f, // skew from 45 to 60 degrees: sqrt(3) - 1
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
      
      angle = 90.0f;
      rvec = new float[]{ -1.0f, 1.0f, 0.0f };  // -45 degree rvec
      tvec = new float[]{ 1.0f, 0.0f, 0.0f };  // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
      Mat4f.set(mexpect, 1.0f,  2.0f,  0.0f,  0.0f,  // skew: from -45 to +45 at height 1, so skew = 2
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));    
      
      angle = 105.0f;
      rvec = new float[]{ -1.0f, 1.0f, 0.0f };  // -45 degree rvec
      tvec = new float[]{ 1.0f, 0.0f, 0.0f };  // X-axis
      m = Mat4f.getSkewMatrix(angle, rvec, tvec);
     
      Mat4f.set(mexpect, 1.0f,  1.0f+rt3,  0.0f,  0.0f,  // skew: from -45 to +60 at height 1, so skew = 1+sqrt(3)
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f));   
   }


   @Test
   public void testGetLookAtMatrix() {
      float[] eyePos    = new float[]{ 0.0f, 0.0f, 0.0f }; // look from origin
      float[] centerPos = new float[]{ 0.0f, 0.0f, -1.0f }; // to -1 on Z
      float[] upVec     = new float[]{ 0.0f, 1.0f, 0.0f }; // up vec as usual: Y = up
      m = Mat4f.getLookAtMatrix(eyePos, centerPos, upVec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,       // we expect the identity
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
      
      
      eyePos    = new float[]{ 0.0f, 0.0f, 2.0f }; // look from z=+2
      centerPos = new float[]{ 0.0f, 0.0f, -1.0f }; // to -1 on Z
      upVec     = new float[]{ 0.0f, 1.0f, 0.0f }; // up vec as usual: Y = up
      m = Mat4f.getLookAtMatrix(eyePos, centerPos, upVec);
      Mat4f.set(mexpect, 1.0f,  0.0f,  0.0f,  0.0f,       // just a translation
                         0.0f,  1.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  1.0f,  -2.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 
     
      eyePos    = new float[]{ 2.0f, 0.0f, 0.0f }; // look from x=+2
      centerPos = new float[]{ 0.0f, 0.0f, 0.0f }; // to origin
      upVec     = new float[]{ 0.0f, 1.0f, 0.0f }; // up vec as usual: Y = up
      m = Mat4f.getLookAtMatrix(eyePos, centerPos, upVec);
      Mat4f.set(mexpect, 0.0f,  0.0f,  -1.0f, -2.0f,      
                         0.0f,  1.0f,  0.0f,  0.0f,
                         1.0f,  0.0f,  0.0f,  0.0f,
                         0.0f,  0.0f,  0.0f,  1.0f);
      assertTrue(Mat4f.epsilonEquals(m, mexpect, 0.0001f)); 


   }

  @Test
   public void testTransposeFloatArrayFloatArray() 
    {
        float m1[]= new float[16];
        for (int i=0;i<16;i++)
        {
            m1[i]=i;
        }
        float m2[]= new float[16];
        Mat4f.transpose(m2,m1);
        Mat4f.transpose(m2);
        for(int i=0;i<16;i++)
        {
            assertTrue(m1[i]==m2[i]);
        }
   }

   
   

   
   
   @Test
   public void testOrthogonalInverse()
   {
       float q[]=new float[4];
       float v[]={1,2,3};
       float m1[]=new float[16];
       float m2[]=new float[16];
       float m[]=new float[16];
       Quat4f.setFromAxisAngle4f(q, 1, 1, -2, 1.2f);
       Mat4f.setFromTRS(m1, v, q,  1);
       Mat4f.rigidInverse(m1, m2);
       Mat4f.mul(m, m1, m2);
       //m * m^-1 = 1
       assertTrue(Mat4f.epsilonEquals(Mat4f.ID,m,0.00001f));
       Mat4f.mul(m, m2, m1);
       //m^-1 * m = 1
       assertTrue(Mat4f.epsilonEquals(Mat4f.ID,m,0.00001f));
       
       Mat4f.set(m2, m1);
       Mat4f.rigidInverse(m1);
       Mat4f.mul(m,m2,m1);
       assertTrue(Mat4f.epsilonEquals(Mat4f.ID,m,0.00001f));
       Mat4f.mul(m,m1,m2);
       assertTrue(Mat4f.epsilonEquals(Mat4f.ID,m,0.00001f));
   }


  



}
