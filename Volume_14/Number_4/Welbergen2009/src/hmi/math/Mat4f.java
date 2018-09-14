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

/**
 * A collection of static methods for 4 X 4 matrices,
 * represented by float arrays of length 16.
 * Matrices are stored in row-major order, i.e.,
 * the first four elements represent the first row,
 * the next four represent the second row etcetera.
 * Note that this deviates from the OpenGL order.
 */
public class Mat4f {

    
    // Constants for accessing 4x4 matrix elements within a float[16] array.
    public static final int m00 = 0;
    public static final int m01 = 1;
    public static final int m02 = 2;
    public static final int m03 = 3;
    public static final int m10 = 4;
    public static final int m11 = 5;
    public static final int m12 = 6;
    public static final int m13 = 7;
    public static final int m20 = 8;
    public static final int m21 = 9;
    public static final int m22 = 10;
    public static final int m23 = 11;
    public static final int m30 = 12;
    public static final int m31 = 13;
    public static final int m32 = 14;
    public static final int m33 = 15;
    
    public static final float degToRadf = (float) (Math.PI/180.0);
    
    /**
     * Returns a new float[16] array with zero components
     */
    public static final float[] getMat4f() {
       return new float[16];
    }
    
    /**
     * Scales the upper left 3X3 part by means of common factor s.
     * Note that this will not cancel scaling factors set earlier,
     * rather, the effect of scale operations is cumulative.
     */
    public static final void scale(float[] m, float s) {
       m[m00] *= s; m[m01] *= s;  m[m02] *= s; 
       m[m10] *= s; m[m11] *= s;  m[m12] *= s; 
       m[m20] *= s; m[m21] *= s;  m[m22] *= s; 
    }  
    
    /**
     * Scales the upper left 3X3 part by means of non-uniform scaling 
     * factors, specified in a Vec3f array. Scaling is performed along the major axes.
     * Note that this will not cancel scaling factors set earlier,
     * rather, the effect of scaling operations is cumulative.
     */
    public static final void nonUniformScale(float[] m, float[] scale) {
       m[m00] *= scale[0]; m[m01] *= scale[1];  m[m02] *= scale[2]; 
       m[m10] *= scale[0]; m[m11] *= scale[1];  m[m12] *= scale[2]; 
       m[m20] *= scale[0]; m[m21] *= scale[1];  m[m22] *= scale[2]; 
    }  
    
    
    /**
     * Allocates a <em>new</em> 4X4 scaling matrix, with scaling
     * factors specified in a length 3 float array.
     */
    public static final float[] getScalingMatrix(float[] s) {
       return new float[] {s[0],0f,0f,0f,  0f,s[1],0f,0f,  0f,0f,s[2],0f,  0f,0f,0f,1f}; 
    } 
    
    
    /**
     * Allocates a <em>new</em> 4X4 translation matrix, with translation
     * vector specified in a length 3 float array.
     */
    public static final float[] getTranslationMatrix(float[] t) {
       return new float[] {1.0f,0f,0f,t[0],  0.0f,1.0f,0f,t[1],  0f,0f,1.0f,t[2],  0f,0f,0f,1f}; 
    } 
    
    /**
     * See getSkewMatrix, with null matrix argument
     */
    public static final float[] getSkewMatrix(float angle, float[] rvec, float[] tvec) {  
        return getSkewMatrix(null, angle, rvec, tvec);
    }
    
    /**
     * Allocates a new skew matrix, specified in Renderman style, by means of a tranlation vector
     * tvec, a rotation vector rvec, and a rotation angle.
     * According to the Renderman specs, shearing is performed in the direction of tvec,
     * where tvec and rvec define the shearing plane. The effect is such that the skewed
     * rvec vector has the specified angle with rvec itself. (Note that the skewed rvec
     * will not have the same length as rvec; despite this, the specs talk abount "rotating"
     * rvec over an angle as specified.) The Collada skew transform for Nodes refers
     * also to the renderman specs, although their example suggests they don't understand it:
     * Rather, they claim that rvec would be the axis of rotation. 
     * The matrix argument should be a Mat4f float array, or it can be null, in which case
     * a float array will be allocated. It is filled with the resulting skewing matrix, and returned
     */
    public static final float[] getSkewMatrix(float[] matrix, float angle, float[] rvec, float[] tvec) {    
       if (matrix == null) matrix = new float[16];
       Mat3f.getSkewMatrix(matrix, angle, rvec, tvec); // gets the 3X3 skewing matrix in the Mat3f part of the matrix array
       convertTo4x4(matrix);   // convert, and add the right column and bottom row for 4X4  
       return matrix; 
    }
    
    /**
     * Allocates a new LookAt matrix, in OpenGL style: from eyePos, look at center, 
     * where upVec is used to infer the direction of the Y-axis (needs not be orthogonal to viewing direction)
     * The viewing direction is considered to be the negative Z-axis,
     * 
     */
    public static final float[] getLookAtMatrix(float[] eyePos, float[] centerPos, float[] upVec) {
       // From eyespace to world space = T_eyePos o M(s, u, -f), where
       // f = normalized (centerPos-eyePos) = -Z axis
       // s = f X normalized upVec = X-axis
       // u = s X f = Y-axis  
       // Invert to obtain lookAt matrix:  (transpose of M) o T_(-eyePos)
       float[] matrix = new float[16];
       float[] f = new float[3];
       Vec3f.sub(f, centerPos, eyePos);
       Vec3f.normalize(f);   // f = normalized viewing direction = -Z axis
       float[] upn = new float[3];
       Vec3f.set(upn, upVec);
       Vec3f.normalize(upn);   // upn = normalized upVec
       float[] s = new float[3];
       Vec3f.cross(s, f, upn); // s = f X normalized upVec = X-axis
       float[] u = new float[3];
       Vec3f.cross(u, s, f);  // u = Y-axis
       Mat4f.set(matrix,
          s[0],  s[1],  s[2], -eyePos[0],
          u[0],  u[1],  u[2], -eyePos[1],
         -f[0], -f[1], -f[2], -eyePos[2],
          0.0f,  0.0f,  0.0f,       1.0f
       );
       return matrix; 
    }
    
    
    /**
     * Creates a new 4X4 matrix from a 3 X 3 matrix, 
     * by adding a right colum and a bottom row, consisting of zero enties.
     * The bottom-right element is set to 1.0, 
     */
    public static float[] from3x3(float[] m3x3) {
       float[] m4x4 = new float[16];
       for (int i=0; i<3; i++) {
          for (int j=0; j<3; j++) {
             m4x4[4*i+j] = m3x3[3*i+j];  
          }
       }
       m4x4[m33] = 1.0f;
       return m4x4; 
    }
    
    /**
     * Converts a 3 X 3 matrix m into a 4 X 4 matrix, 
     * by adding a right colum and a bottom row, consisting of zero enties.
     * The bottom-right element is set to 1.0.
     * This is done "in place", therefore m should be a float array of length 16(!) 
     */
    public static void convertTo4x4(float[] m) {
       m[m22] = m[Mat3f.m22];
       m[m21] = m[Mat3f.m21];
       m[m20] = m[Mat3f.m20];
       m[m12] = m[Mat3f.m12];
       m[m11] = m[Mat3f.m11];
       m[m10] = m[Mat3f.m10];
       // m00, m01, m02 positions coincide, so no copy needed
       m[m03] = m[m13] = m[m23] = 0.0f; // zero rightmost column
       m[m30] = m[m31] = m[m32] = 0.0f; // zero bottom row        
       m[m33] = 1.0f;
    }
    
    
    /**
     * Copies a 4X4 matrix src into matrix dst
     */
    public static final void set(float[] dst, int dstIndex, float[] src, int srcIndex ) {
       System.arraycopy(src, srcIndex, dst, dstIndex, 16); 
    }
    
    /**
     * Copies a 4X4 matrix src into matrix dst
     */
    public static final void set(float[] dst, float[] src) {
       System.arraycopy(src, 0, dst, 0, 16); 
    }
    
    /**
     * Sets the 4x4 matrix in dst
     */
    public static final void set(float[] dst, float src00, float src01, float src02, float src03, 
    		                                  float src10, float src11, float src12, float src13, 
    		                                  float src20, float src21, float src22, float src23,
    		                                  float src30, float src31, float src32, float src33)    
    {
    	dst[m00]=src00; dst[m10]=src10; dst[m20]=src20;dst[m30]=src30;
    	dst[m01]=src01; dst[m11]=src11; dst[m21]=src21;dst[m31]=src31;
    	dst[m02]=src02; dst[m12]=src12; dst[m22]=src22;dst[m32]=src32;
    	dst[m03]=src03; dst[m13]=src13; dst[m23]=src23;dst[m33]=src33;
    }
    
    
    /**
     * Sets all matrix component, from a rotation Quat4f  q, a rotation center Vec3f c,
     * and a uniform scale.
     */
    public static final void setFromTRCS(float[] m, float[] t, float[] q, float[] c, float uscale) {
       setFromTRCSVec3f(m, t, q, c, new float[]{uscale, uscale, uscale});
    }
    
    /**
     * Sets all matrix component, from a rotation Quat4f  q, a rotation center Vec3f c,
     * scaling factors in the form of a Vec3f, and a Vec3f translation.
     * Scaling is performed along the main axes. 
     */
    public static final void setFromTRCSVec3f(float[] m, float[] t, float[] q, float[] c, float[] s) {
        // set rotation/scale part in upper-left 3 X 3 matrix
        m[m00] =  s[0] * ( 1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m01] =  s[1] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m02] =  s[2] * ( 2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
                          
        m[m10] =  s[0] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m11] =  s[1] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m12] =  s[2] * (-2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
                          
        m[m20] =  s[0] * (-2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
        m[m21] =  s[1] * ( 2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
        m[m22] =  s[2] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y]);
        
        // set translation, including center contribution
        m[m03] =  t[0] + c[0] - m[m00]*c[0] - m[m01]*c[1] - m[m02]*c[2];
        m[m13] =  t[1] + c[1] - m[m10]*c[0] - m[m11]*c[1] - m[m12]*c[2];
        m[m23] =  t[2] + c[2] - m[m20]*c[0] - m[m21]*c[1] - m[m22]*c[2];
        
        // set last row:
        m[m30] = 0.0f;   m[m31] = 0.0f;    m[m32] = 0.0f;    m[m33] = 1.0f; 
    }
    
    
    /**
     * Sets all matrix component, from a rotation Quat4f  q, 
     * scaling matrix in the form of a Mat3f, and a Vec3f translation. 
     */
    public static final void setFromTRSMat3f(float[] m, float[] t, float[] q, float[] smatrix) {
        //(rr0, rr1, rr2) = row-i from the unscaled rotation matrix
        // calculate first row of rotation matrix
        float rr0 = 1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        float rr1 = 2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z];
        float rr2 = 2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        
        // matmult with smatrix yields first row of m
        m[m00] = smatrix[Mat3f.m00]*rr0 + smatrix[Mat3f.m10]*rr1 + smatrix[Mat3f.m20]*rr2;
        m[m01] = smatrix[Mat3f.m01]*rr0 + smatrix[Mat3f.m11]*rr1 + smatrix[Mat3f.m21]*rr2;
        m[m02] = smatrix[Mat3f.m02]*rr0 + smatrix[Mat3f.m12]*rr1 + smatrix[Mat3f.m22]*rr2; 
        
        // calculate second row of rotation matrix                  
        rr0 =  2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z];
        rr1 =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        rr2 = -2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        
        // matmult with smatrix yields second row of m
        m[m10] = smatrix[Mat3f.m00]*rr0 + smatrix[Mat3f.m10]*rr1 + smatrix[Mat3f.m20]*rr2;
        m[m11] = smatrix[Mat3f.m01]*rr0 + smatrix[Mat3f.m11]*rr1 + smatrix[Mat3f.m21]*rr2;
        m[m12] = smatrix[Mat3f.m02]*rr0 + smatrix[Mat3f.m12]*rr1 + smatrix[Mat3f.m22]*rr2;
          
        // calculate third row of rotation matrix                    
        rr0 = -2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        rr1 =  2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        rr2 =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y];
        
        // matmult with smatrix yields third row of m
        m[m20] = smatrix[Mat3f.m00]*rr0 + smatrix[Mat3f.m10]*rr1 + smatrix[Mat3f.m20]*rr2;
        m[m21] = smatrix[Mat3f.m01]*rr0 + smatrix[Mat3f.m11]*rr1 + smatrix[Mat3f.m21]*rr2;
        m[m22] = smatrix[Mat3f.m02]*rr0 + smatrix[Mat3f.m12]*rr1 + smatrix[Mat3f.m22]*rr2;
        
        // set translation column
        m[m03] =  t[0];
        m[m13] =  t[1];
        m[m23] =  t[2];
        
        // set last row:
        m[m30] = 0.0f;   m[m31] = 0.0f;    m[m32] = 0.0f;    m[m33] = 1.0f; 
    }
    
    
    
    /**
     * Sets all matrix component, from a translation Vec3f t, rotation Quat4f q, 
     * and (non uniform) scaling factors in the form of a Vec3f.
     * Scaling is performed along the main axes.
     */
    public static final void setFromTRSVec3f(float[] m, float[] t, float[] q, float[] s) {
        // set rotation/scale part in upper-left 3 X 3 matrix
        m[m00] =  s[0] * ( 1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m01] =  s[1] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m02] =  s[2] * ( 2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
                          
        m[m10] =  s[0] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m11] =  s[1] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m12] =  s[2] * (-2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
                          
        m[m20] =  s[0] * (-2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
        m[m21] =  s[1] * ( 2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
        m[m22] =  s[2] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y]);
        
        // set translation column
        m[m03] =  t[0];  m[m13] =  t[1];   m[m23] =  t[2];
        
        // set bottom row:
        m[m30] = 0.0f;   m[m31] = 0.0f;    m[m32] = 0.0f;    m[m33] = 1.0f; 
    }
    
    
    /**
     * Sets all matrix component, from a translation Vec3f t, a rotation Quat4f  q, 
     * and a uniform scaling float factor uscale.
     */
    public static final void setFromTRS(float[] m, float[] t, float[] q, float uscale) {
      // set rotation/scale part in upper-left 3 X 3 matrix
        m[m00] =  uscale * ( 1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m01] =  uscale * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m02] =  uscale * ( 2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
               
        m[m10] =  uscale * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m11] =  uscale * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m12] =  uscale * (-2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
                  
        m[m20] =  uscale * (-2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
        m[m21] =  uscale * ( 2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
        m[m22] =  uscale * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y]);
        
        // set translation column
        m[m03] =  t[0];  m[m13] =  t[1];   m[m23] =  t[2];
        
        // set bottom row:
        m[m30] = 0.0f;   m[m31] = 0.0f;    m[m32] = 0.0f;    m[m33] = 1.0f; 
    }
    
    
    /**
     * Sets all matrix component, from a translation Vec3f t and a rotation Quat4f  q. 
     */
    public static final void setFromTR(float[] m, float[] t, float[] q) {
       // set rotation/scale part in upper-left 3 X 3 matrix
        m[m00] =  1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        m[m01] =  2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z];
        m[m02] =  2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        
        m[m10] =  2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z];
        m[m11] =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        m[m12] = -2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        
        m[m20] = -2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        m[m21] =  2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        m[m22] =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y];
        
        // set translation column
        m[m03] =  t[0];  m[m13] =  t[1];   m[m23] =  t[2];
        
        // set bottom row:
        m[m30] = 0.0f;   m[m31] = 0.0f;    m[m32] = 0.0f;    m[m33] = 1.0f; 
    }
    
    
    /**
     * Sets the rotation part of a 4X4 (or 3X4) matrix m, i.e. the upper left 3X3 part, 
     * from a Quat4f quaternion q and a Vec3f scaling array in the form of a Vec3f array.
     * The remaining parts are not modified. 
     */
    public static final void setRotationScaleVec3f(float[] m, float[] q, float[] s) {
        m[m00] =  s[0] * ( 1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m01] =  s[1] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m02] =  s[2] * ( 2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
                         
        m[m10] =  s[0] * ( 2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z]);
        m[m11] =  s[1] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z]);
        m[m12] =  s[2] * (-2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
                         
        m[m20] =  s[0] * (-2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z]);
        m[m21] =  s[1] * ( 2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z]);
        m[m22] =  s[2] * ( 1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y]);
    }
    
    
    /**
     * Sets the rotation part of a 4X4 (or 3X4) matrix m, i.e. the upper left 3X3 part, 
     * from a Quat4f quaternion q.
     * The remaining parts are not modified. 
     */
    public static final void setRotation(float[] m, int mIndex, float[] q, int qIndex) {
        m[m00+mIndex] =  1.0f - 2.0f*q[Quat4f.y+qIndex]*q[Quat4f.y+qIndex] - 2.0f*q[Quat4f.z+qIndex]*q[Quat4f.z+qIndex];
        m[m01+mIndex] =  2.0f*q[Quat4f.x+qIndex]*q[Quat4f.y+qIndex] - 2.0f*q[Quat4f.s+qIndex]*q[Quat4f.z+qIndex];
        m[m02+mIndex] =  2.0f*q[Quat4f.s+qIndex]*q[Quat4f.y+qIndex] + 2.0f*q[Quat4f.x+qIndex]*q[Quat4f.z+qIndex];
        
        m[m10+mIndex] =  2.0f*q[Quat4f.x+qIndex]*q[Quat4f.y+qIndex] + 2.0f*q[Quat4f.s+qIndex]*q[Quat4f.z+qIndex];
        m[m11+mIndex] =  1.0f - 2.0f*q[Quat4f.x+qIndex]*q[Quat4f.x+qIndex] - 2.0f*q[Quat4f.z+qIndex]*q[Quat4f.z+qIndex];
        m[m12+mIndex] = -2.0f*q[Quat4f.s+qIndex]*q[Quat4f.x+qIndex] + 2.0f*q[Quat4f.y+qIndex]*q[Quat4f.z+qIndex];
        
        m[m20+mIndex] = -2.0f*q[Quat4f.s+qIndex]*q[Quat4f.y+qIndex] + 2.0f*q[Quat4f.x+qIndex]*q[Quat4f.z+qIndex];
        m[m21+mIndex] =  2.0f*q[Quat4f.s+qIndex]*q[Quat4f.x+qIndex] + 2.0f*q[Quat4f.y+qIndex]*q[Quat4f.z+qIndex];
        m[m22+mIndex] =  1.0f - 2.0f*q[Quat4f.x+qIndex]*q[Quat4f.x+qIndex] - 2.0f*q[Quat4f.y+qIndex]*q[Quat4f.y+qIndex];
    }
    
    /**
     * Sets the rotation part of a 4X4 (or 3X4) matrix m, i.e. the upper left 3X3 part, 
     * from a Quat4f quaternion q.
     * The remaining parts are not modified. 
     */
    public static final void setRotation(float[] m, float[] q) {
        m[m00] =  1.0f - 2.0f*q[Quat4f.y]*q[Quat4f.y] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        m[m01] =  2.0f*q[Quat4f.x]*q[Quat4f.y] - 2.0f*q[Quat4f.s]*q[Quat4f.z];
        m[m02] =  2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        
        m[m10] =  2.0f*q[Quat4f.x]*q[Quat4f.y] + 2.0f*q[Quat4f.s]*q[Quat4f.z];
        m[m11] =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.z]*q[Quat4f.z];
        m[m12] = -2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        
        m[m20] = -2.0f*q[Quat4f.s]*q[Quat4f.y] + 2.0f*q[Quat4f.x]*q[Quat4f.z];
        m[m21] =  2.0f*q[Quat4f.s]*q[Quat4f.x] + 2.0f*q[Quat4f.y]*q[Quat4f.z];
        m[m22] =  1.0f - 2.0f*q[Quat4f.x]*q[Quat4f.x] - 2.0f*q[Quat4f.y]*q[Quat4f.y];
    }
    
   
    
    /**
     * Sets the rotation part of a 4X4 or 3X4 matrix m, i.e. the upper left 3X3 part, from an rotation axis and an angle,
     * specified in degrees, not radians. The axis need not have length 1 for this operation.
     */
    public static final void setRotationFromAxisAngleDegrees(float[] m, float[] axis, float degrees) {
       float[] aa = new float[3];
       Vec3f.set(aa, axis);
       Vec3f.normalize(aa);
       setRotationFromAxisAngle(m, aa, degrees * degToRadf );
    }
    
    /**
     * Sets the rotation part of a 4X4 or 3X4 matrix m, i.e. the upper left 3X3 part, from a axis-and-angle array,
     * of length 4. The angle is specified in radians, the axis should have length 1.
     * The remaining parts are not modified. 
     */
    public static final void setRotationFromAxisAngle4f(float[] m, float[] axisangle) {
       setRotationFromAxisAngle(m, axisangle, axisangle[3]);
    }
    
    /**
     * Sets the rotation part of a 4X4 or 3X4 matrix m, i.e. the upper left 3X3 part, from an axis aray, of length 3,
     * and an float angle. The angle is specified in radians, the axis should have length 1.
     * The remaining parts are not modified. 
     */
    public static final void setRotationFromAxisAngle(float[] m, float[] axis,float angle) 
    {
    	//axis-angle to quaternion
        float qs = (float) Math.cos(angle/2.0);
        float sn = (float) Math.sin(angle/2.0);
        float qx = axis[0] * sn;
        float qy = axis[1] * sn;
        float qz = axis[2] * sn;
        
    	  m[m00] =  (float) (1.0 - 2.0*qy*qy - 2.0*qz*qz);
        m[m01] =  (float) (2.0*qx*qy - 2.0*qs*qz);
        m[m02] =  (float) (2.0*qs*qy + 2.0*qx*qz);
        
        m[m10] =  (float) (2.0*qx*qy + 2.0*qs*qz);
        m[m11] =  (float) (1.0 - 2.0*qx*qx - 2.0*qz*qz);
        m[m12] =  (float) (-2.0*qs*qx + 2.0*qy*qz);
        
        m[m20] =  (float) (-2.0*qs*qy + 2.0*qx*qz);
        m[m21] =  (float) ( 2.0*qs*qx + 2.0*qy*qz);
        m[m22] =  (float) (1.0 - 2.0*qx*qx - 2.0*qy*qy);
    }
    
    /**
     * Sets the translation vector column for a 4X4 (or 3X4) matrix m,
     * i.e. the last 3X1 column, from a translation Vec3f vector t.
     * The remaining parts are not modified.
     */
    public static final void setTranslation(float[] m, float[] t) {
        m[m03] = t[0];
        m[m13] = t[1];
        m[m23] = t[2];
    }
    

    /**
     * Retrieves the translation vector column for a 4X4 or 3X4 matrix m,
     * i.e. the last 3X1 column, to a translation Vec3f vector t.
     */
    public static final void getTranslation(float[] t, float[] m) {
       t[0] = m[m03];
       t[1] = m[m13];
       t[2] = m[m23];
    }



    /**
     * Resets the 4X4 matrix to zero.
     */
    public static final void setZero(float[] m) {
        for (int i=0; i<16; i++) m[i] = 0.0f;
    }
    
    /**
     * Resets the 4X4 matrix to the identity matrix.
     */
    public static final void setIdentity(float[] m) {
        for (int i=1; i<15; i++) m[i] = 0.0f;
        m[m00] =  m[m11] = m[m22] = m[m33] = 1.0f;
    }
    
    /**
     * Allocates a <em>new</em> 4X4 matrix, initialized to the identity matrix.
     */
    public static final float[] getIdentity() {
       return new float[] {1f,0f,0f,0f, 0f,1f,0f,0f, 0f,0f,1f,0f, 0f,0f,0f,1f}; 
    } 

    /**
     * Checks whether some matrix is actually the identity matrix.
     * This checks for exact identity.
     */
    public static final boolean isIdentity(float[] m) {
        return (
           m[m00] == 1.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 1.0 && m[m02] == 0.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 1.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 1.0 
        );
    }

    /**
     * Checks whether some matrix is actually the zero matrix.
     * This checks for exact identity.
     */
    public static final boolean isZero(float[] m) {
        return (
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 0.0 && 
           m[m00] == 0.0 && m[m01] == 0.0 && m[m02] == 0.0 && m[m03] == 0.0 
        );
    }


    /**
     * Sets the element m(i,j) from a (row-major) 4X4 matrix m to a
     * specified float value.
     */
    public static final void setElement(float[] m, int i, int j, float value ) {
        m[4*i+j] = value;
    }
    
    /**
     * Gets the float value of matrix element m(i,j), form a row-major order 4X4 matrix m.
     */
    public static final float getElement(float[] m, int i, int j) {
        return m[4*i+j];
    }
    
    /**
     * Copies a matrix row with index i from a 4X4 matrix m.
     * The result is copied to a Vec4 array row.
     */
    public static final void getRow(float[] m, int i, float[] row) {
        int offset = 4*i;
        row[0] = m[offset];
        row[1] = m[offset+1];
        row[2] = m[offset+2];
        row[3] = m[offset+3];
    }
    
    /**
     * Copies a matrix column with index j from a 4X4 matrix m.
     * The result is copied to a Vec4 array col.
     */
    public static final void getColumn(float[] m, int j, float[] col) {
        col[0] = m[j];
        col[1] = m[j+4];
        col[2] = m[j+8];
        col[3] = m[j+12];
    }
    
    /**
     * Multiplies Mat4f matrix dest with Mat4f matrix A and stores the result back in dest.
     * dest = dest * A
     */
    public static final void mul(float[] dest, float[] A) {
       mul(dest, dest, A);
    }
    
    /**
     * Multiplies Mat4f matrix A with Mat4f matrix B and stores the result in Mat4f matrix dest.
     * The dest array is allowed to be aliased with A and/or B:
     * dest = A * B
     */
    public static final void mul(float[] dest, float[] A, float[] B) {
        float mt00 = A[m00]*B[m00] + A[m01]*B[m10] + A[m02]*B[m20] + A[m03]*B[m30];
        float mt01 = A[m00]*B[m01] + A[m01]*B[m11] + A[m02]*B[m21] + A[m03]*B[m31];
        float mt02 = A[m00]*B[m02] + A[m01]*B[m12] + A[m02]*B[m22] + A[m03]*B[m32];
        float mt03 = A[m00]*B[m03] + A[m01]*B[m13] + A[m02]*B[m23] + A[m03]*B[m33];
                                                                   
        float mt10 = A[m10]*B[m00] + A[m11]*B[m10] + A[m12]*B[m20] + A[m13]*B[m30];
        float mt11 = A[m10]*B[m01] + A[m11]*B[m11] + A[m12]*B[m21] + A[m13]*B[m31];
        float mt12 = A[m10]*B[m02] + A[m11]*B[m12] + A[m12]*B[m22] + A[m13]*B[m32];
        float mt13 = A[m10]*B[m03] + A[m11]*B[m13] + A[m12]*B[m23] + A[m13]*B[m33];
                                                                     
        float mt20 = A[m20]*B[m00] + A[m21]*B[m10] + A[m22]*B[m20] + A[m23]*B[m30];
        float mt21 = A[m20]*B[m01] + A[m21]*B[m11] + A[m22]*B[m21] + A[m23]*B[m31];
        float mt22 = A[m20]*B[m02] + A[m21]*B[m12] + A[m22]*B[m22] + A[m23]*B[m32];
        float mt23 = A[m20]*B[m03] + A[m21]*B[m13] + A[m22]*B[m23] + A[m23]*B[m33];
                                                                   
        float mt30 = A[m30]*B[m00] + A[m31]*B[m10] + A[m32]*B[m20] + A[m33]*B[m30];
        float mt31 = A[m30]*B[m01] + A[m31]*B[m11] + A[m32]*B[m21] + A[m33]*B[m31];
        float mt32 = A[m30]*B[m02] + A[m31]*B[m12] + A[m32]*B[m22] + A[m33]*B[m32];
        float mt33 = A[m30]*B[m03] + A[m31]*B[m13] + A[m32]*B[m23] + A[m33]*B[m33];
                                                                    
        dest[m00] = mt00;  dest[m01] = mt01;  dest[m02] = mt02;  dest[m03] = mt03;
        dest[m10] = mt10;  dest[m11] = mt11;  dest[m12] = mt12;  dest[m13] = mt13;
        dest[m20] = mt20;  dest[m21] = mt21;  dest[m22] = mt22;  dest[m23] = mt23;
        dest[m30] = mt30;  dest[m31] = mt31;  dest[m32] = mt32;  dest[m33] = mt33;		
    }
    
    /**
     * Multiplies two Mat4f matrices, assuming that the fourth ro is of the form (0, 0, 0, 1). 
     * Whether the fourth row is actually present or not is not important;
     * The Mat4f coding stores these in the last four elements of a float[16] array. 
     * This method does not read nor write these elements. 
     * The dest array is allowed to be aliased with A and/or B:
     * dest = A * B
     */
    public static final void mul3x4(float[] dest, float[] A, float[] B) {
        float mt00 = A[m00]*B[m00] + A[m01]*B[m10] + A[m02]*B[m20];
        float mt01 = A[m00]*B[m01] + A[m01]*B[m11] + A[m02]*B[m21];
        float mt02 = A[m00]*B[m02] + A[m01]*B[m12] + A[m02]*B[m22];
        float mt03 = A[m00]*B[m03] + A[m01]*B[m13] + A[m02]*B[m23] + A[m03];
        
        float mt10 = A[m10]*B[m00] + A[m11]*B[m10] + A[m12]*B[m20];
        float mt11 = A[m10]*B[m01] + A[m11]*B[m11] + A[m12]*B[m21];
        float mt12 = A[m10]*B[m02] + A[m11]*B[m12] + A[m12]*B[m22];
        float mt13 = A[m10]*B[m03] + A[m11]*B[m13] + A[m12]*B[m23] + A[m13];
                                                                    
        float mt20 = A[m20]*B[m00] + A[m21]*B[m10] + A[m22]*B[m20];
        float mt21 = A[m20]*B[m01] + A[m21]*B[m11] + A[m22]*B[m21];
        float mt22 = A[m20]*B[m02] + A[m21]*B[m12] + A[m22]*B[m22];
        float mt23 = A[m20]*B[m03] + A[m21]*B[m13] + A[m22]*B[m23] + A[m23];
                                                                    
        dest[m00] = mt00;  dest[m01] = mt01;  dest[m02] = mt02;  dest[m03] = mt03;
        dest[m10] = mt10;  dest[m11] = mt11;  dest[m12] = mt12;  dest[m13] = mt13;
        dest[m20] = mt20;  dest[m21] = mt21;  dest[m22] = mt22;  dest[m23] = mt23;	
    }

    /**
     * Equivalent to mul(dest, dest, A). That, multiplies the 3x4 dest matrix on the right with the
     * 3x4 matrix A, treating the fourth row of dest and A as being (0, 0, 0, 1)
     */
    public static final void mul3x4(float[] dest, float[] A){ 
       mul3x4(dest, dest, A);
    }

    /**
     * Transforms a Vec4 vector src, and puts the result in 
     * vector dest. The latter is allowed to be aliased to src.
     * The matrix, as well as the two vectors start at offsets, specified by mIndex, destIndex,
     * and srcIndex.
     */
    public static void transformVec4f(float[]m, int mIndex, float[] dest, int destIndex, float[] src, int srcIndex) {
        float vx = m[mIndex+m00]*src[srcIndex]+m[mIndex+m01]*src[srcIndex+1]+m[mIndex+m02]*src[srcIndex+2]+m[mIndex+m03]*src[srcIndex+3];
        float vy = m[mIndex+m10]*src[srcIndex]+m[mIndex+m11]*src[srcIndex+1]+m[mIndex+m12]*src[srcIndex+2]+m[mIndex+m13]*src[srcIndex+3];
        float vz = m[mIndex+m20]*src[srcIndex]+m[mIndex+m21]*src[srcIndex+1]+m[mIndex+m22]*src[srcIndex+2]+m[mIndex+m23]*src[srcIndex+3];
        float vw = m[mIndex+m30]*src[srcIndex]+m[mIndex+m31]*src[srcIndex+1]+m[mIndex+m32]*src[srcIndex+2]+m[mIndex+m33]*src[srcIndex+3]; 
        dest[destIndex]  =vx;
        dest[destIndex+1]=vy;
        dest[destIndex+2]=vz;
        dest[destIndex+3]=vw;
    }
    
    /**
     * Transforms a Vec4 vector src, and puts the result in 
     * vector dest. The latter is allowed to be aliased to src.
     */
    public static void transformVec4f(float[]m, float[] dest, float[] src) {
        float vx = m[m00]*src[0]+m[m01]*src[1]+m[m02]*src[2]+m[m03]*src[3];
        float vy = m[m10]*src[0]+m[m11]*src[1]+m[m12]*src[2]+m[m13]*src[3];
        float vz = m[m20]*src[0]+m[m21]*src[1]+m[m22]*src[2]+m[m23]*src[3];
        float vw = m[m30]*src[0]+m[m31]*src[1]+m[m32]*src[2]+m[m33]*src[3];
        dest[0]=vx;
        dest[1]=vy;
        dest[2]=vz;
        dest[3]=vw;
    }
    
    /**
     * Transforms a Vec4 vector dezt in place
     */
    public static void transformVec4f(float[]m, float[] dest) {
       transformVec4f(m, dest, dest);
    }
    
    /**
     * transforms a Vec3 vector and stores the result in the same vector
     */
    public static void transformVec3f(float[]m,float[] dest) {
    	transformVec3f(m,dest,dest);	
    }
    
    /**
     * Transforms a Vec3 (NB!) vector src, and puts the result in 
     * vector dest. The latter is allowed to be aliased to src.
     * In effect the matrix is treated like an 3X3 matrix m33 plus
     * an extra 3X1 colum vector t. (The last matrix row is ignored altogether)
     * The result: dest = m33 * src + t
     */
    public static void transformVec3f(float[]m, float[] dest, float[] src) {
        float vx = m[m00]*src[0]+m[m01]*src[1]+m[m02]*src[2]+m[m03];
        float vy = m[m10]*src[0]+m[m11]*src[1]+m[m12]*src[2]+m[m13];
        float vz = m[m20]*src[0]+m[m21]*src[1]+m[m22]*src[2]+m[m23];
        dest[0]=vx;
        dest[1]=vy;
        dest[2]=vz;
    }
    
    /**
     * Transforms a Vec3 (NB!) vector dest starting at the specified destIndex offset, 
     * In effect the matrix is treated like an 3X3 matrix m33 plus
     * an extra 3X1 colum vector t. (The last matrix row is ignored altogether)
     * The result: dest = m33 * dest + t
     */
    public static void transformVec3f(float[]m, float[] dest, int destIndex) {
        float vx = m[m00]*dest[destIndex]+m[m01]*dest[destIndex+1]+m[m02]*dest[destIndex+2]+m[m03];
        float vy = m[m10]*dest[destIndex]+m[m11]*dest[destIndex+1]+m[m12]*dest[destIndex+2]+m[m13];
        float vz = m[m20]*dest[destIndex]+m[m21]*dest[destIndex+1]+m[m22]*dest[destIndex+2]+m[m23];
        dest[destIndex]=vx;
        dest[destIndex+1]=vy;
        dest[destIndex+2]=vz;
    }
    
    
    /**
     * Sets matrix dest to the transpose of matrix m.
     * dest and m can be the same matrix.
     */
    public static final void transpose(float[] dest, float[] m) {
        float tmp;
        dest[m00]=m[m00];
        dest[m11]=m[m11];
        dest[m22]=m[m22];
        dest[m33]=m[m33];
        tmp = m[m01]; dest[m01] = m[m10]; dest[m10] = tmp; 
        tmp = m[m02]; dest[m02] = m[m20]; dest[m20] = tmp; 
        tmp = m[m03]; dest[m03] = m[m30]; dest[m30] = tmp; 
        tmp = m[m12]; dest[m12] = m[m21]; dest[m21] = tmp; 
        tmp = m[m13]; dest[m13] = m[m31]; dest[m31] = tmp; 
        tmp = m[m23]; dest[m23] = m[m32]; dest[m32] = tmp; 
    }
    
    /**
     * Transposes matrix m.
     */
    public static final void transpose(float[] m) {
        float tmp;        
        tmp = m[m01]; m[m01] = m[m10]; m[m10] = tmp; 
        tmp = m[m02]; m[m02] = m[m20]; m[m20] = tmp; 
        tmp = m[m03]; m[m03] = m[m30]; m[m30] = tmp; 
        tmp = m[m12]; m[m12] = m[m21]; m[m21] = tmp; 
        tmp = m[m13]; m[m13] = m[m31]; m[m31] = tmp; 
        tmp = m[m23]; m[m23] = m[m32]; m[m32] = tmp; 
    }
    
    /**
     * Efficient method for calculating the inverse of a rigid transform.
     * This assumes that m has the form T(t) o R, where T(t) translates
     * over vector t, and where R is a pure rotation, without scaling. 
     * In this case, the inverse is T(R'(-t)) o R', where R' is the transpose of R.
     */
    public static final void rigidInverse(float[]m, float dest[]) {
        //transpose of the 3x3 rotation matrix part R
        float tmp;
        dest[m00]=m[m00];
        dest[m11]=m[m11];
        dest[m22]=m[m22];
        tmp = m[m01]; dest[m01] = m[m10]; dest[m10] = tmp; 
        tmp = m[m02]; dest[m02] = m[m20]; dest[m20] = tmp; 
        tmp = m[m12]; dest[m12] = m[m21]; dest[m21] = tmp;
        
        //vector part is -R^T*v
        float vx = m[m03];
        float vy = m[m13];
        float vz = m[m23];
        dest[m03] = -dest[m00]*vx-dest[m01]*vy-dest[m02]*vz;        
        dest[m13] = -dest[m10]*vx-dest[m11]*vy-dest[m12]*vz;
        dest[m23] = -dest[m20]*vx-dest[m21]*vy-dest[m22]*vz;
        dest[m30] = 0;
        dest[m31] = 0;
        dest[m32] = 0;
        dest[m33] = 1;    
    }
    
    /**
     * Like rigidInverse(m, m). So it is assumed that m is rigid:
     * a translation and rotation, but no scaling/skewing. 
     */
    public static final void rigidInverse(float[]m) {
        rigidInverse(m,m);
    }
    
    /**
     * The 4X4 identity matrix.
     */
    public static final float[] ID = new float[] {1f, 0f, 0f, 0f,
                                                  0f, 1f, 0f, 0f,
                                                  0f, 0f, 1f, 0f,
                                                  0f, 0f, 0f, 1f};
    
    
    /**
     * Tests for (strict) equality of matrix components.
     */
    public static final boolean equals(float[] a, float[] b) {
        float diff;
        for(int i=0;i<16;i++)
        {
           diff = a[i] - b[i];
           if(Float.isNaN(diff)) return false;
           if(diff != 0.0f) return false;
       }
       return true;  
    }
    
    
    /**
     * Tests for equality of matrix components within epsilon.
     */
    public static final boolean epsilonEquals(float[] a, float[] b, float epsilon) {
        float diff;
        for (int i=0; i<16; i++) {
           diff = a[i] - b[i];
           if (Float.isNaN(diff)) return false;
           if (Math.abs(diff) > epsilon) return false;
       }
       return true;  
    }
    
    /**
     * Produces a String representation of a 4 X 4 matrix, suitable for
     * printing, debugging etcetera
     */
    public static String toString(float[]m ) {
       return toString(m, 0);
    }


    /**
     * Produces a String representation of Mat4f matrix m, taking into account
     * tab spaces at the beginning of every newline. 
     * Matrix elements within eps from 1.0 or -1.0
     * are represented by 1.0 or -1.0, elements with absolute value
     * < eps will be presented as 0.0 values.
     */
    public static String toString(float[] m, int tab, float eps) {
      StringBuilder buf = new StringBuilder();
      for (int i=0; i<4; i++) {
         buf.append('\n');
         for (int t=0; t<tab; t++) buf.append(' ');
         for (int j=0; j<4; j++) {
            float mval = m[4*i+j];
            if (Math.abs(mval) < eps) {
               buf.append(" 0.0");
            } else if (Math.abs(1.0f-mval) < eps) {
               buf.append(" 1.0");
            } else if (Math.abs(-1.0f-mval) < eps) {
               buf.append("-1.0");
            } else {
               buf.append(mval);
            }
            buf.append("  ");  
         }  
      }
      return buf.toString();  
   }


   /**
     * Produces a String representation of Mat4f matrix m, taking into account
     * tab spaces at the beginning of every newline. 
     * Matrix elements within eps from 1.0 or -1.0
     * are represented by 1.0 or -1.0, elements with absolute value
     * < eps will be presented as 0.0 values.
     */
    public static String toString(float[] m, int tab) {
       StringBuilder buf = new StringBuilder();
      for (int i=0; i<4; i++) {
         buf.append('\n');
         for (int t=0; t<tab; t++) buf.append(' ');
         for (int j=0; j<4; j++) {
            buf.append(m[4*i+j]);
            buf.append("  ");  
         }  
      }
      return buf.toString();  
   }

   
    /**
     * Checks the bottom row of a 4X4 matrix. Returns true if it is
     * not equal to 0, 0, 0, 1, as is the case for projection matrices.
     */
    public static final  boolean isProjective(float[] matrix) {
       return matrix[m30] != 0.0f ||  matrix[m31] != 0.0f || matrix[m32] != 0.0f || matrix[m33] != 1.0f;
    }
 
}
