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
 * A collection of static methods for 3 X 3 matrices,
 * represented by float arrays of length 9.
 * Matrices are stored in row-major order, i.e.,
 * the first three elements represent the first row,
 * the next three represent the second row etcetera.
 * Note that this deviates from the OpenGL order.
 */
public class Mat3f {

    // Constants for accessing 3X3 matrix elements within a float[9] array.
    public static final int m00 = 0;
    public static final int m01 = 1;
    public static final int m02 = 2;
   
    public static final int m10 = 3;
    public static final int m11 = 4;
    public static final int m12 = 5;
    
    public static final int m20 = 6;
    public static final int m21 = 7;
    public static final int m22 = 8;
    
    
    /**
     * Returns a new float[9] array with zero components
     */
    public static final float[] getMat3f() {
       return new float[9];
    }
   
    
    /**
     * Allocates a <em>new</em> 3X3 scaling matrix, with scaling
     * factors specified in a length 3 float array.
     */
    public static final float[] getScalingMatrix(float[] s) {
       return new float[] {s[0],0f,0f,  0f,s[1],0f,  0f,0f,s[2]}; 
    } 
    
    /**
     * Scales the matrix m by means of factor s.
     */
    public static final void scale(float[] m, float s) {
       m[m00] *= s; m[m01] *= s;  m[m02] *= s; 
       m[m10] *= s; m[m11] *= s;  m[m12] *= s; 
       m[m20] *= s; m[m21] *= s;  m[m22] *= s; 
    }  
    
    /**
     * Scales the matrix m by means of factor s.
     */
    public static final void scale(float[] m, int iIndex, float s) {
       m[iIndex+m00] *= s; m[iIndex+m01] *= s;  m[iIndex+m02] *= s; 
       m[iIndex+m10] *= s; m[iIndex+m11] *= s;  m[iIndex+m12] *= s; 
       m[iIndex+m20] *= s; m[iIndex+m21] *= s;  m[iIndex+m22] *= s; 
    }  
    
    /**
     * Scales the matrix m by means of factor s.
     */
    public static final void scale(float[] m, int iIndex, double s) {
       m[iIndex+m00] *= s; m[iIndex+m01] *= s;  m[iIndex+m02] *= s; 
       m[iIndex+m10] *= s; m[iIndex+m11] *= s;  m[iIndex+m12] *= s; 
       m[iIndex+m20] *= s; m[iIndex+m21] *= s;  m[iIndex+m22] *= s; 
    }  
    
    /**
     * Scales the matrix m by means of factor s.
     */
    public static final void scale(float[] m, double s) {
       m[m00] *= s; m[m01] *= s;  m[m02] *= s; 
       m[m10] *= s; m[m11] *= s;  m[m12] *= s; 
       m[m20] *= s; m[m21] *= s;  m[m22] *= s; 
    } 
    
    /**
     * Copies a 3X3 matrix src into matrix dst
     */
    public static final void set(float[] dst, int dstIndex, float[] src, int srcIndex ) {
        for (int i=0; i<9; i++) dst[dstIndex+i] = src[srcIndex+i];
    }
    
    /**
     * Sets the matrix from 9 float values;
     */
    public static final void set(float[] dst, float src00, float src01, float src02, float src10, float src11, float src12, float src20, float src21, float src22)
    {
        dst[m00]=src00; dst[m01]=src01; dst[m02]=src02;
        dst[m10]=src10; dst[m11]=src11; dst[m12]=src12;
        dst[m20]=src20; dst[m21]=src21; dst[m22]=src22;
    }
    
    /**
     * Sets the matrix from 9 float values;
     */
    public static final void set(float[] dst, int dIndex, float src00, float src01, float src02, float src10, float src11, float src12, float src20, float src21, float src22)
    {
        dst[dIndex+m00]=src00; dst[dIndex+m01]=src01; dst[dIndex+m02]=src02;
        dst[dIndex+m10]=src10; dst[dIndex+m11]=src11; dst[dIndex+m12]=src12;
        dst[dIndex+m20]=src20; dst[dIndex+m21]=src21; dst[dIndex+m22]=src22;
    }
    
    /**
     * Copies a 3X3 matrix src into matrix dst
     */
    public static final void set(float[] dst, float[] src) {
        for (int i=0; i<9; i++) dst[i] = src[i];
    }
    
    /**
     * Sets a 3X3 matrix from a unit quaternion and scale factor.
     */
    public static final void setFromQuatScale(float[] m, float[] q, float s) {
        m[0]  = (float) (s *(1.0 - 2.0*q[Quat4f.y]*q[Quat4f.y] - 2.0*q[Quat4f.z]*q[Quat4f.z]));
        m[1]  = (float) (s *( 2.0*q[Quat4f.x]*q[Quat4f.y] - 2.0*q[Quat4f.s]*q[Quat4f.z]));
        m[2]  = (float) (s *( 2.0*q[Quat4f.s]*q[Quat4f.y] + 2.0*q[Quat4f.x]*q[Quat4f.z]));
   
        m[3]  = (float) (s *( 2.0*q[Quat4f.x]*q[Quat4f.y] + 2.0*q[Quat4f.s]*q[Quat4f.z]));
        m[4]  = (float) (s *(1.0 - 2.0*q[Quat4f.x]*q[Quat4f.x] - 2.0*q[Quat4f.z]*q[Quat4f.z]));
        m[5]  = (float) (s *( 2.0*q[Quat4f.y]*q[Quat4f.z] -2.0*q[Quat4f.s]*q[Quat4f.x] ));
        
        m[6]  = (float) (s *( 2.0*q[Quat4f.x]*q[Quat4f.z]-2.0*q[Quat4f.s]*q[Quat4f.y]));
        m[7]  = (float) (s *( 2.0*q[Quat4f.s]*q[Quat4f.x] + 2.0*q[Quat4f.y]*q[Quat4f.z]));
        m[8]  = (float) (s *(1.0 - 2.0*q[Quat4f.x]*q[Quat4f.x] - 2.0*q[Quat4f.y]*q[Quat4f.y]));         
    }
    
    /**
     * Sets a 3X3 matrix from a unit quaternion and scale factor.
     */
    public static final void setFromQuatScale(float[] m, int mIndex, float[] q, int qIndex, float s) {
        m[mIndex]    = (float) (s *(1.0 - 2.0*q[qIndex+Quat4f.y]*q[qIndex+Quat4f.y] - 2.0*q[qIndex+Quat4f.z]*q[qIndex+Quat4f.z]));
        m[mIndex+1]  = (float) (s *( 2.0*q[qIndex+Quat4f.x]*q[qIndex+Quat4f.y] - 2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.z]));
        m[mIndex+2]  = (float) (s *( 2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.y] + 2.0*q[qIndex+Quat4f.x]*q[qIndex+Quat4f.z]));
   
        m[mIndex+3]  = (float) (s *( 2.0*q[qIndex+Quat4f.x]*q[qIndex+Quat4f.y] + 2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.z]));
        m[mIndex+4]  = (float) (s *(1.0 - 2.0*q[Quat4f.x]*q[qIndex+Quat4f.x] - 2.0*q[qIndex+Quat4f.z]*q[qIndex+Quat4f.z]));
        m[mIndex+5]  = (float) (s *( 2.0*q[qIndex+Quat4f.y]*q[qIndex+Quat4f.z] -2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.x] ));
        
        m[mIndex+6]  = (float) (s *( 2.0*q[qIndex+Quat4f.x]*q[qIndex+Quat4f.z]-2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.y]));
        m[mIndex+7]  = (float) (s *( 2.0*q[qIndex+Quat4f.s]*q[qIndex+Quat4f.x] + 2.0*q[qIndex+Quat4f.y]*q[qIndex+Quat4f.z]));
        m[mIndex+8]  = (float) (s *(1.0 - 2.0*q[qIndex+Quat4f.x]*q[qIndex+Quat4f.x] - 2.0*q[qIndex+Quat4f.y]*q[qIndex+Quat4f.y]));         
    }
    
    /**
     * Sets a 3X3 matrix from an axis (float[3]), angle and scale factor.
     * The axis need not have length 1.
     */
    public static final void setFromAxisAngleScale(float[] m, float[] axis, float angle, float scale) 
    {
        //axis-angle to quaternion
        
        float a0 = axis[0]; float a1 = axis[1]; float a2 = axis[2]; 
        float axisLenSq = a0*a0 + a1*a1 + a2*a2;
        
        float qs = (float) Math.cos(angle/2.0);
        float sn = (float) Math.sin(angle/2.0);
        if (Math.abs(axisLenSq-1.0f) > 0.001f) {
           sn *= (float) (1.0/Math.sqrt(axisLenSq));
        } 
        float qx = a0 * sn;
        float qy = a1 * sn;
        float qz = a2 * sn;

        //quaternion to matrix
        m[0]  = (float) (scale *(1.0 - 2.0*qy*qy - 2.0*qz*qz));
        m[1]  = (float) (scale *( 2.0*qx*qy - 2.0*qs*qz));
        m[2]  = (float) (scale *( 2.0*qs*qy + 2.0*qx*qz));

        m[3]  = (float) (scale *( 2.0*qx*qy + 2.0*qs*qz));
        m[4]  = (float) (scale *(1.0 - 2.0*qx*qx - 2.0*qz*qz));
        m[5]  = (float) (scale *(2.0*qy*qz -2.0*qs*qx));

        m[6]  = (float) (scale *(2.0*qx*qz -2.0*qs*qy ));
        m[7]  = (float) (scale *( 2.0*qs*qx + 2.0*qy*qz));
        m[8]  = (float) (scale *(1.0 - 2.0*qx*qx - 2.0*qy*qy));
    }
    
    /**
     * Sets a 3X3 matrix from a axis angle aa and scale factor.
     * The first three components of aa define the axis, the fouth one is the angle.
     
     */    
    public static final void setFromAxisAngleScale(float[] m, float[] aa, float scale) {
        //axis-angle to quaternion
        float a0 = aa[0]; float a1 = aa[1]; float a2 = aa[2]; float halfangle = aa[3]/2.0f;
        float axisLenSq = a0*a0 + a1*a1 + a2*a2;
        
        float qs = (float) Math.cos(halfangle);
        float sn = (float) Math.sin(halfangle);
        if (Math.abs(axisLenSq-1.0f) > 0.001f) {
         sn *= (float) (1.0/Math.sqrt(axisLenSq));
        }       
        float qx = a0 * sn;
        float qy = a1 * sn;
        float qz = a2 * sn;

        //quaternion to matrix
        m[0]  = (float) (scale *(1.0 - 2.0*qy*qy - 2.0*qz*qz));
        m[1]  = (float) (scale *( 2.0*qx*qy - 2.0*qs*qz));
        m[2]  = (float) (scale *( 2.0*qs*qy + 2.0*qx*qz));

        m[3]  = (float) (scale *( 2.0*qx*qy + 2.0*qs*qz));
        m[4]  = (float) (scale *(1.0 - 2.0*qx*qx - 2.0*qz*qz));
        m[5]  = (float) (scale *(2.0*qy*qz -2.0*qs*qx));

        m[6]  = (float) (scale *(2.0*qx*qz -2.0*qs*qy ));
        m[7]  = (float) (scale *( 2.0*qs*qx + 2.0*qy*qz));
        m[8]  = (float) (scale *(1.0 - 2.0*qx*qx - 2.0*qy*qy));
    }
    
    
    /**
     * Resets the 3X3 matrix to zero
     */
    public static final void setZero(float[] m) {
        for (int i=0; i<9; i++) m[i] = 0.0f;
    }
    
    /**
     * Resets the 3X3 matrix to the identity matrix.
     */
    public static final void setIdentity(float[] m) {
        for (int i=1; i<8; i++) m[i] = 0.0f;
        m[m00] = m[m11] = m[m22] = 1.0f;
    }

    /**
     * Allocates a <em>new</em> 3X3 matrix, initialized to the identity matrix.
     */
    public static final float[] getIdentity() {
       return new float[] {1f,0f,0f, 0f,1f,0f, 0f,0f,1f}; 
    } 


    /**
     * Sets the element m(i,j) from a (row-major) 3X3 matrix m to a
     * specified float value.
     */
    public static final void setElement(float[] m, int i, int j, float value ) {
        m[3*i+j] = value;
    }
    
    /**
     * Gets the float value of matrix element m(i,j), form a row-major order 3X3 matrix m.
     */
    public static final float getElement(float[] m, int i, int j) {
        return m[3*i+j];
    }
    
    /**
     * Copies a matrix row with index i from a 3X3 matrix m.
     * The result is copied to a Vec3 array row.
     */
    public static final void getRow(float[] m, int i, float[] row) {
        int offset = 3*i;
        row[0] = m[offset];
        row[1] = m[offset+1];
        row[2] = m[offset+2];
    }
    
    /**
     * Copies a matrix column with index j from a 3X3 matrix m.
     * The result is copied to a Vec3 array col.
     */
    public static final void getColumn(float[] m, int j, float[] col) {
        col[0] = m[j];
        col[1] = m[j+3];
        col[2] = m[j+6];
    }
    
    /**
     * Multiplies A with B and stores the result in dest.
     * The dest array is allowed to be aliased with A and/or B:
     * dest = A * B
     */
    public static final void mul(float[] dest, int dIndex, float[] A,int aIndex, float[] B, int bIndex) 
    {
        float mt00 = A[aIndex+m00]*B[bIndex+m00] + A[aIndex+m01]*B[bIndex+m10] + A[aIndex+m02]*B[bIndex+m20];
        float mt01 = A[aIndex+m00]*B[bIndex+m01] + A[aIndex+m01]*B[bIndex+m11] + A[aIndex+m02]*B[bIndex+m21];
        float mt02 = A[aIndex+m00]*B[bIndex+m02] + A[aIndex+m01]*B[bIndex+m12] + A[aIndex+m02]*B[bIndex+m22];
        
        float mt10 = A[aIndex+m10]*B[bIndex+m00] + A[aIndex+m11]*B[bIndex+m10] + A[aIndex+m12]*B[bIndex+m20];
        float mt11 = A[aIndex+m10]*B[bIndex+m01] + A[aIndex+m11]*B[bIndex+m11] + A[aIndex+m12]*B[bIndex+m21];
        float mt12 = A[aIndex+m10]*B[bIndex+m02] + A[aIndex+m11]*B[bIndex+m12] + A[aIndex+m12]*B[bIndex+m22];
        
        float mt20 = A[aIndex+m20]*B[bIndex+m00] + A[aIndex+m21]*B[bIndex+m10] + A[aIndex+m22]*B[bIndex+m20];
        float mt21 = A[aIndex+m20]*B[bIndex+m01] + A[aIndex+m21]*B[bIndex+m11] + A[aIndex+m22]*B[bIndex+m21];
        float mt22 = A[aIndex+m20]*B[bIndex+m02] + A[aIndex+m21]*B[bIndex+m12] + A[aIndex+m22]*B[bIndex+m22];

        dest[dIndex+m00] = mt00;  dest[dIndex+m01] = mt01;  dest[dIndex+m02] = mt02;
        dest[dIndex+m10] = mt10;  dest[dIndex+m11] = mt11;  dest[dIndex+m12] = mt12;
        dest[dIndex+m20] = mt20;  dest[dIndex+m21] = mt21;  dest[dIndex+m22] = mt22;     
    }
    
    /**
     * Multiplies A with B^T and stores the result in dest.
     * The dest array is allowed to be aliased with A and/or B:
     * dest = A * B^T
     */
    public static final void mulTransposeRight(float[] dest, float[] A, float[] B) 
    {
        float bt00,bt01,bt02,bt10,bt11,bt12,bt20,bt21,bt22;
        bt00 = B[m00];
        bt11 = B[m11];
        bt22 = B[m22];
        bt10 = B[m01]; bt01 = B[m10]; 
        bt20 = B[m02]; bt02 = B[m20]; 
        bt21 = B[m12]; bt12 = B[m21];
        
        float mt00 = A[m00]*bt00 + A[m01]*bt10 + A[m02]*bt20;
        float mt01 = A[m00]*bt01 + A[m01]*bt11 + A[m02]*bt21;
        float mt02 = A[m00]*bt02 + A[m01]*bt12 + A[m02]*bt22;
        
        float mt10 = A[m10]*bt00 + A[m11] * bt10 + A[m12] * bt20;
        float mt11 = A[m10]*bt01 + A[m11] * bt11 + A[m12] * bt21;
        float mt12 = A[m10]*bt02 + A[m11] * bt12 + A[m12] * bt22;
        
        float mt20 = A[m20]*bt00 + A[m21] * bt10 + A[m22]*bt20;
        float mt21 = A[m20]*bt01 + A[m21] * bt11 + A[m22]*bt21;
        float mt22 = A[m20]*bt02 + A[m21] * bt12 + A[m22]*bt22;

        dest[m00] = mt00;  dest[m01] = mt01;  dest[m02] = mt02;
        dest[m10] = mt10;  dest[m11] = mt11;  dest[m12] = mt12;
        dest[m20] = mt20;  dest[m21] = mt21;  dest[m22] = mt22;     
    }
    
    /**
     * Multiplies A with B and stores the result in dest.
     * The dest array is allowed to be aliased with A and/or B:
     * dest = A * B
     */
    public static final void mul(float[] dest, float[] A, float[] B) {
        float mt00 = A[m00]*B[m00] + A[m01]*B[m10] + A[m02]*B[m20];
        float mt01 = A[m00]*B[m01] + A[m01]*B[m11] + A[m02]*B[m21];
        float mt02 = A[m00]*B[m02] + A[m01]*B[m12] + A[m02]*B[m22];
        
        float mt10 = A[m10]*B[m00] + A[m11]*B[m10] + A[m12]*B[m20];
        float mt11 = A[m10]*B[m01] + A[m11]*B[m11] + A[m12]*B[m21];
        float mt12 = A[m10]*B[m02] + A[m11]*B[m12] + A[m12]*B[m22];
        
        float mt20 = A[m20]*B[m00] + A[m21]*B[m10] + A[m22]*B[m20];
        float mt21 = A[m20]*B[m01] + A[m21]*B[m11] + A[m22]*B[m21];
        float mt22 = A[m20]*B[m02] + A[m21]*B[m12] + A[m22]*B[m22];

        dest[m00] = mt00;  dest[m01] = mt01;  dest[m02] = mt02;
        dest[m10] = mt10;  dest[m11] = mt11;  dest[m12] = mt12;
        dest[m20] = mt20;  dest[m21] = mt21;  dest[m22] = mt22;    	
    }
    
    
    /**
     * Multiplies dest with m and stores the result in dest:
     * dest = dest * m
     */
    public static final void mul(float[] dest, float [] m) {
       mul(dest, dest, m);
    }
    
    /**
     * Multiplies dest with m and stores the result in dest:
     * dest = dest * m
     */
    public static final void mul(float[] dest, int destIndex, float [] m, int mIndex) 
    {
       mul(dest, destIndex, dest, destIndex, m, mIndex);
    }
    
    /**
     * Transforms a 3 float src, and puts the result in 
     * vector dest. 
     * The dst vector and matrix start at a offset, specified by destIndex and mIndex      
     */
    public static void transform(float[]m, int mIndex, float[] dest, int destIndex, float srcx, float srcy, float srcz) 
    {
        dest[destIndex]   = m[mIndex+m00]*srcx+m[mIndex+m01]*srcy+m[mIndex+m02]*srcz;
        dest[destIndex+1] = m[mIndex+m10]*srcx+m[mIndex+m11]*srcy+m[mIndex+m12]*srcz;
        dest[destIndex+2] = m[mIndex+m20]*srcx+m[mIndex+m21]*srcy+m[mIndex+m22]*srcz;        
    }
    
    /**
     * Transforms a 3 float src, and puts the result in 
     * vector dest. 
     * The dst vector starts at a offset, specified by destIndex
     */
    public static void transform(float[]m, float[] dest, int destIndex, float srcx, float srcy, float srcz) 
    {
        dest[destIndex]   = m[m00]*srcx+m[m01]*srcy+m[m02]*srcz;
        dest[destIndex+1] = m[m10]*srcx+m[m11]*srcy+m[m12]*srcz;
        dest[destIndex+2] = m[m20]*srcx+m[m21]*srcy+m[m22]*srcz;        
    }
    
    /**
     * Transforms a Vec3 vector src, and puts the result in 
     * vector dest. The latler is allowed to be aliased to src.
     * The matrix, as well as the two vectors start at offsets, specified by mIndex, destIndex,
     * and srcIndex.
     */
    public static void transform(float[]m, int mIndex, float[] dest, int destIndex, float[] src, int srcIndex) {
        float vx = m[mIndex+m00]*src[srcIndex]+m[mIndex+m01]*src[srcIndex+1]+m[mIndex+m02]*src[srcIndex+2];
        float vy = m[mIndex+m10]*src[srcIndex]+m[mIndex+m11]*src[srcIndex+1]+m[mIndex+m12]*src[srcIndex+2];
        float vz = m[mIndex+m20]*src[srcIndex]+m[mIndex+m21]*src[srcIndex+1]+m[mIndex+m22]*src[srcIndex+2];
        dest[destIndex]=vx;
        dest[destIndex+1]=vy;
        dest[destIndex+2]=vz;
    }
    
    
    /**
     * Transforms a Vec3 vector dst, and puts the result back in 
     * vector dst. 
     * The matrix, as well as the  vector start at offsets, specified by mIndex, dstIndex,
     */
    public static void transform(float[]m, int mIndex, float[] dst, int dstIndex) {
        float vx = m[mIndex+m00]*dst[dstIndex]+m[mIndex+m01]*dst[dstIndex+1]+m[mIndex+m02]*dst[dstIndex+2];
        float vy = m[mIndex+m10]*dst[dstIndex]+m[mIndex+m11]*dst[dstIndex+1]+m[mIndex+m12]*dst[dstIndex+2];
        float vz = m[mIndex+m20]*dst[dstIndex]+m[mIndex+m21]*dst[dstIndex+1]+m[mIndex+m22]*dst[dstIndex+2];
        dst[dstIndex]=vx;
        dst[dstIndex+1]=vy;
        dst[dstIndex+2]=vz;
    }
    
    
    
    /**
     * Transforms a Vec3 vector src, and puts the result in 
     * vector dest. The latter is allowed to be aliased to src.
     */
    public static void transform(float[]m, float[] dest, float[] src) {
        float vx = m[m00]*src[0]+m[m01]*src[1]+m[m02]*src[2];
        float vy = m[m10]*src[0]+m[m11]*src[1]+m[m12]*src[2];
        float vz = m[m20]*src[0]+m[m21]*src[1]+m[m22]*src[2];
        dest[0]=vx;
        dest[1]=vy;
        dest[2]=vz;
    }
    
    
    /**
     * Transforms a Vec3 vector dst, and puts the result back in 
     * vector dst. 
     */
    public static void transform(float[]m, float[] dst) {
        float vx = m[m00]*dst[0]+m[m01]*dst[1]+m[m02]*dst[2];
        float vy = m[m10]*dst[0]+m[m11]*dst[1]+m[m12]*dst[2];
        float vz = m[m20]*dst[0]+m[m21]*dst[1]+m[m22]*dst[2];
        dst[0]=vx;
        dst[1]=vy;
        dst[2]=vz;
    }
    
    
     /**
     * Tests for equality of matrix components.
     */
    public static final boolean equals(float[] a, float[] b)
    {
        float diff;
        for(int i=0;i<9;i++)
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
    public static final boolean epsilonEquals(float[] a, float[] b, float epsilon)
    {
        float diff;
        for(int i=0;i<9;i++)
        {
           diff = a[i] - b[i];
           if(Float.isNaN(diff)) return false;
           if((diff<0 ? -diff : diff) > epsilon) return false;
       }
       return true;  
    }
    
    /**
     * Transforms a Vec3 with the transpose of matrix m 
     */
    public static final void transformTranspose(float[] m, float[] dst, float[] src) 
    {
        float vx = m[m00]*src[0]+m[m10]*src[1]+m[m20]*src[2];
        float vy = m[m01]*src[0]+m[m11]*src[1]+m[m21]*src[2];
        float vz = m[m02]*src[0]+m[m12]*src[1]+m[m22]*src[2];
        dst[0] = vx;
        dst[1] = vy;
        dst[2] = vz;
    }
    
    
    /**
     * Transforms a Vec3 with the matrix transpose 
     */
    public static final void transformTranspose(float[] m,int mIndex,float[] dst, int dstIndex, float[] src, int srcIndex) 
    {
        float vx = m[mIndex+m00]*src[srcIndex]+m[mIndex+m10]*src[srcIndex+1]+m[mIndex+m20]*src[srcIndex+2];
        float vy = m[mIndex+m01]*src[srcIndex]+m[mIndex+m11]*src[srcIndex+1]+m[mIndex+m21]*src[srcIndex+2];
        float vz = m[mIndex+m02]*src[srcIndex]+m[mIndex+m12]*src[srcIndex+1]+m[mIndex+m22]*src[srcIndex+2];
        dst[dstIndex]   = vx;
        dst[dstIndex+1] = vy;
        dst[dstIndex+2] = vz; 
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
        tmp = m[m01]; dest[m01] = m[m10]; dest[m10] = tmp; 
        tmp = m[m02]; dest[m02] = m[m20]; dest[m20] = tmp; 
        tmp = m[m12]; dest[m12] = m[m21]; dest[m21] = tmp; 
    }
    
    /**
     * Transposes matrix m.
     */
    public static final void transpose(float[] m) {
        float tmp;
        tmp = m[m01]; m[m01] = m[m10]; m[m10] = tmp; 
        tmp = m[m02]; m[m02] = m[m20]; m[m20] = tmp; 
        tmp = m[m12]; m[m12] = m[m21]; m[m21] = tmp; 
    }
    
    /**
     * dest += a
     */
    public static final void add(float dest[], float a[]) {
       for(int i=0;i<9;i++)  dest[i]+=a[i];      
    }
    
    /**
     * dest -= a
     */
    public static final void sub(float dest[], float a[]) {
       for(int i=0;i<9;i++)   dest[i]-=a[i];        
    }
    
    
    /**
     * dest += b
     */
    public static final void add(float dest[], int destIndex, float a[], int aIndex) {
       for(int i=0;i<9;i++)  dest[i+destIndex]+=a[i+aIndex];     
    }
    
    /**
     * dest = a+b
     */
    public static final void add(float dest[],float a[],float b[]) {
       for(int i=0;i<9;i++) dest[i]=a[i]+b[i];      
    }
    
    /**
     * dest = a-b
     */
    public static final void sub(float dest[],float a[],float b[]) {
       for(int i=0;i<9;i++)  dest[i]=a[i]-b[i];     
    }
    
    /**
     * dest = a+b
     */
    public static final void add(float dest[],int destIndex, float a[], int aIndex, float b[], int bIndex) {
       for(int i=0;i<9;i++)  dest[i+destIndex]=a[i+aIndex]+b[i+bIndex];       
    }
    
    /**
     * Determines the determinant of m.
     */
    public static double det(float m[]) {
        return  m[m00]*(m[m11]*m[m22]-m[m12]*m[m21])+
                m[m01]*(m[m12]*m[m20]-m[m10]*m[m22])+
                m[m02]*(m[m10]*m[m21]-m[m11]*m[m20]);
    }
    
    /**
     * Sets the dest matrix to the transpose of the adjugate (i.e "classical adjoint" matrix of m)
     * It is always defined, even when m is not invertible.
     * If m is invertible, i.e. det(m) != 0, then adjugate_transpose(m) = (transpose invert(m)) * det(m)
     * Returns the deteminant of m.
     */
    public static float adjugate_transpose(float[] dest, float[] m) {
      Vec3f.cross(dest, 0, m, 3, m, 6); // dest-row-0 = m-row_1 X m-row-2
      Vec3f.cross(dest, 3, m, 6, m, 0); // dest-row-1 = m-row_2 X m-row-0
      Vec3f.cross(dest, 6, m, 0, m, 3); // dest-row-2 = m-row_0 X m-row-1
      return m[m00]*dest[m00] +  m[m01]*dest[m01] +  m[m02]*dest[m02];
    }
    
    /**
     * Sets the dest matrix to the transpose of the adjugate of the rotation/scaling 3x3 part
     * of the 4x4 matrix m. 
     * Returns the deteminant
     */
    protected static float adjugate_transposeMat4f(float[] dest, float[] m) {
      Vec3f.cross(dest, 0, m, 4, m, 8); // dest-row-0 = m-row_1 X m-row-2
      Vec3f.cross(dest, 3, m, 8, m, 0); // dest-row-1 = m-row_2 X m-row-0
      Vec3f.cross(dest, 6, m, 0, m, 4); // dest-row-2 = m-row_0 X m-row-1
      return m[m00]*dest[m00] +  m[m01]*dest[m01] +  m[m02]*dest[m02];
    }
    
    
    
    /**
     * Sets the dest matrix to the adjugate (i.e "classical adjoint") matrix of m)
     * It is always defined, even when m is not invertible.
     * If m is invertible, i.e. det(m) != 0, then adjugate(m) = invert(m) * det(m)
     * Return the determinant det(m)
     */
    public static float adjugate(float[] dest, float[] m) {
       dest[m00]= m[m22]*m[m11]-m[m21]*m[m12]; dest[m01]=-m[m22]*m[m01]+m[m21]*m[m02];  dest[m02]= m[m12]*m[m01]-m[m11]*m[m02];
       dest[m10]=-m[m22]*m[m10]+m[m20]*m[m12]; dest[m11]= m[m22]*m[m00]-m[m20]*m[m02];  dest[m12]=-m[m12]*m[m00]+m[m10]*m[m02];
       dest[m20]= m[m21]*m[m10]-m[m20]*m[m11]; dest[m21]=-m[m21]*m[m00]+m[m20]*m[m01];  dest[m22]= m[m11]*m[m00]-m[m10]*m[m01];
       return m[m00]*dest[m00] +  m[m01]*dest[m10] +  m[m02]*dest[m20];
    }
    
    /**
     * Inverts matrix m and returns the determinant of m.
     * If the latter is equal to zero, the adjugate is returned in dest, and zero is returned.
     * Hint: Use the more efficient transpose method if the matrix is known to be orthogonal
     */
    public static final float invert(float[] dest, float[] m) {
       float det = adjugate(dest, m);
       if (det == 0.0f) {
         //throw new IllegalArgumentException("Mat3f.invert: singular matrix");
         return 0.0f;
       }
       scale(dest, 1.0f/det);
       return det;
    }
    
    
    /**
     * Sets dest to the transpose of the inverted m matrix, and returns the determinant of m.
     * If the latter is equal to zero, the adjugate_transpose is returned in dest, and zero is returned.
     * Hint: Use the more efficient transpose method if the matrix is known to be orthogonal
     */
    public static final float invert_transpose(float[] dest, float[] m) {
       float det = adjugate_transpose(dest, m);
       if (det == 0.0f) {
         //throw new IllegalArgumentException("Mat3f.invert: singular matrix");
         return 0.0f;
       }
       scale(dest, 1.0f/det);
       return det;
    }
    
    /**
     * Sets dest to the 3x3 matrix that is the inverse-transpose of the rotation/scaling part of
     * the 4x4 matrix m
     */
    public static final float invert_transposeMat4f(float[] dest3x3, float[] m4x4) {
       float det = adjugate_transposeMat4f(dest3x3, m4x4);
       if (det == 0.0f) {
         return 0.0f;
       }
       scale(dest3x3, 1.0f/det);
       return det;
    }
    
    
    /**
     * Return the norm_1 of matrix m: the sum of the absolute values of all matrix elements.
     */
    public static float norm_1(float[] m) {
       float sum = 0.0f;
       for (int i=0; i<9; i++) {
          sum += (m[i] < 0.0f) ? -m[i] : m[i];  
       }
       return sum;   
    }
    
    /**
     * Return the norm-2 of matrix m: the square root of the sum of the squares of all elements.
     * (The standard euclidean norm)
     */
    public static float norm_2(float[] m) {
       float sum = 0.0f;
       for (int i=0; i<9; i++) {
          sum += m[i]*m[i];
       }
       return (float) Math.sqrt(sum);   
    }
    
    /**
     * Return the max norm of matrix m: the max absolute value of the matrix elements.
     */
    public static float norm_inf(float[] m) {
       float max = 0.0f;
       for (int i=0; i<9; i++) {
          float mx = (m[i] < 0.0f) ? -m[i] : m[i];
          if (mx > max) max = mx;
       }
       return max;   
    }
    
    /**
     * Allocates a new 3 X 3 matrix, containing
     * a copy of the upper-left 3 X 3 matrix from an 4 X 4 matrix.
     */
    public static float[] from4x4(float[] m4x4) {
       float[] m3x3 = new float[9];
       for (int i=0; i<3; i++) {
          for (int j=0; j<3; j++) {
             m3x3[3*i+j] = m4x4[4*i+j];  
          }
       }
       return m3x3; 
    }
    
    /**
     * Converts a 4 X 4 matrix, stored in a length 16(!) float array m
     * into a 3 X 3 matrix, by dropping the rightmost column, and the bottom row.
     *  This is done "in place", so m remains
     * a length 16 float array, where the first 9 array elements
     * now contain the Mat3f matrix.
     */
    public static void convertTo3x3(float[] m) {
       // m00, m01, m02 positions coincide, so no copy needed
       m[m10] = m[Mat4f.m10];
       m[m11] = m[Mat4f.m11];
       m[m12] = m[Mat4f.m12];
       m[m20] = m[Mat4f.m20];
       m[m21] = m[Mat4f.m21];
       m[m22] = m[Mat4f.m22];
    }
    
    
    /**
     * Determines whether some 3 X 3 matrix is in diagonal form.
     * By definition, this is the case when all off-diagonal elements
     * have absolute value less than epsilon
     */
    public static boolean isDiagonal(float[] m, float epsilon) {
       return (
       Math.abs(m[m01]) < epsilon && Math.abs(m[m02]) < epsilon && 
       Math.abs(m[m10]) < epsilon && Math.abs(m[m12]) < epsilon &&
       Math.abs(m[m20]) < epsilon && Math.abs(m[m21]) < epsilon  );
    }
    
    
    /**
     * Returns the diagonal in a Vec3f array
     */
    public static void getDiagonal(float[] matrix3f, float[] vec3f) {
        vec3f[0] = matrix3f[m00]; 
        vec3f[1] = matrix3f[m11]; 
        vec3f[2] = matrix3f[m22];   
    } 
    
    /**
     * Sets the diagonal elements in a 3 X 3 matrix from a Vec3f array.
     * The remaining matrix elements are not modified.
     */
    public static void setDiagonal(float[] matrix3f, float[] vec3f) {
       matrix3f[m00] = vec3f[0];
       matrix3f[m11] = vec3f[1];
       matrix3f[m22] = vec3f[2];
    } 
    
    
    /**
     * Determines whether some 3 X 3 matrix is symmetric.
     * By definition, this is the case when Math.abs (mij - mji) < epsilon,
     * for all i != j
     */
    public static boolean isSymmetric(float[] m, float epsilon) {
       return (
       Math.abs(m[m01] - m[m10]) < epsilon && 
       Math.abs(m[m02] - m[m20]) < epsilon && 
       Math.abs(m[m12] - m[m21]) < epsilon );    
    }
    
    
    public static final float TOL = (float) 1.0e-6;
    
    public enum ScalingType {IDENTITY, UNIFORM, ALIGNED, SKEW, UNDEFINED}
    
    /**
     * Returns the scaling type for a vector of scale factors.
     * This is one of ScalingType.IDENTITY, ScalingType.UNIFORM, or ScalingType.ALIGNED.
     */
    public static final ScalingType getScalingTypeVec3f(float[] scaleVec) {
       if (scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2]) return ScalingType.ALIGNED;
       if (scaleVec[0] == 1.0f) return ScalingType.IDENTITY;
       return ScalingType.UNIFORM;
    }
    
    /**
     * Performs a polar decomposition of matrix M into factor Q and S: M = Q S
     * Q is orthogonal, S is symmetric. In essence, Q is the rotation part,
     * S is the scaling matrix. Note that the latter can scale along axes that
     * are not aligned with the x-y-z axes, in which case S is also called skewing or shearing.
     * Returns the scaling type.
     * The epsilon parameter determines the threshold for smoothing the scaling matrix.
     */
    public static ScalingType polar_decompose(float[] M, float[] Q, float[] S, float epsilon) {
       float[] Mk = new float[9];     // k-th iteration for Q
       float[] MkInvT = new float[9]; // k-th iteration of inverse transpose of Mk
       float[] Ek = new float[9];     // error estimation for k-th iteration
       set(Mk, M);
       float n_1 =norm_1(Mk);        
       float n_inf = norm_inf(Mk);
       float n_inv_1, n_inv_inf, n_e_1, det;
       do {
          det = invert_transpose(MkInvT, Mk);  
          if (det == 0.0f) return ScalingType.UNDEFINED;
          n_inv_1 = norm_1(MkInvT); 
          n_inv_inf = norm_inf(MkInvT);
          float r = (n_inv_1 * n_inv_inf) / (n_1 * n_inf);
          float gamma  = (float) Math.sqrt(Math.sqrt(r));     
          float g1 = 0.5f * gamma;
          float g2 = 0.5f / gamma;
          set(Ek, Mk);
          // Mk = g1 * Mk + g2 * MkInvT
          scale(Mk, g1);
          scale(MkInvT, g2);
          add(Mk, MkInvT);
          sub(Ek, Mk);
          n_e_1 = norm_1(Ek);
          n_1 = norm_1(Mk);
          n_inf = norm_inf(Mk);         
       } while (n_e_1 > TOL * n_1);
       set(Q, Mk);
       transpose(Mk);
       mul(S, Mk, M);
       transpose(Ek, S); // reuse Ek for tranpose of S
       add(S, Ek);
       scale(S, 0.5f);
       float eps = 0.0001f;
       smooth(S, eps);
       return getScalingType(S);
    }
    
    
    /**
     * Matrix elements close to 0.0, 1.0, or -1.0 are rounded towards those values,
     * provided the difference is less than eps
     */
    public static void smooth(float[] m, float eps) {
        for (int i=0; i<9; i++) {
           if (Math.abs(m[i]) < eps) {
              m[i] = 0.0f;
           } else if (Math.abs(m[i] - 1.0f) < eps) {
             m[i] = 1.0f;
           } else if (Math.abs(m[i] + 1.0f) < eps) {
             m[i] = 1.0f;
           }
        }
    }
    
    /**
     * Determines the scaling type of a matrix
     */
    public static ScalingType getScalingType(float[] m) {
       // first check all off-diagonal elements:
       if (   m[m01] != 0.0f || m[m02] != 0.0f || m[m12] != 0.0f || m[m10] != 0.0f ||m[m20] != 0.0f || m[m21] != 0.0f) {
          return ScalingType.SKEW;
       }
       // off-diagonal elements all zero, check for uniformity:
       if ( (Math.abs(m[m00] - m[m11]) > 0.0f) || (Math.abs(m[m11] - m[m22]) > 0.0f) ) {
          return ScalingType.ALIGNED; // non-uniform scaling, but aligned with the axes.
       }   
       if (Math.abs(m[m00] - 1.0f) > 0.0f) {
          return ScalingType.UNIFORM; // uniform scaling, with some non-unit scaling factor
       }     
       return ScalingType.IDENTITY; // No scaling at all
    }
    
    /**
     * Sets the skew matrix 
     *   0  -vz   vy
     *  vz    0  -vx
     * -vy   vx    0
     * from a vector
     */
    public static final void skew(float[]m, float v[])
    {
        m[m00] = 0;     m[m01]=-v[2];   m[m02]=v[1];
        m[m10] = v[2];  m[m11]=0;       m[m12]=-v[0];
        m[m20] = -v[1]; m[m21]=v[0];    m[m22]=0;
    }
    
    /**
     * Sets the skew matrix 
     *   0  -vz   vy
     *  vz    0  -vx
     * -vy   vx    0
     * from a vector
     */
    public static final void skew(float[]m, float vx, float vy, float vz)
    {
        m[m00] = 0;     m[m01]=-vz;   m[m02]=vy;
        m[m10] = vz;    m[m11]=0;     m[m12]=-vx;
        m[m20] = -vy;   m[m21]=vx;    m[m22]=0;
    }
    
    /**
     * Sets the skew matrix 
     *   0  -vz   vy
     *  vz    0  -vx
     * -vy   vx    0
     * from a vector
     */
    public static final void skew(float[]m, int mIndex, float v[],int vIndex)
    {
        m[mIndex+m00] = 0;              m[mIndex+m01]=-v[2+vIndex];   m[mIndex+m02]=v[1+vIndex];
        m[mIndex+m10] = v[vIndex+2];    m[mIndex+m11]=0;              m[mIndex+m12]=-v[vIndex];
        m[mIndex+m20] = -v[vIndex+1];   m[mIndex+m21]=v[vIndex];      m[mIndex+m22]=0;
    }
    
    /**
     * Like getSkewMatrix with null matrix argument
     */
    public static final float[] getSkewMatrix(float angle, float[] rvec, float[] tvec) {
      return getSkewMatrix(null, angle, rvec, tvec);
    } 
    
    /**
     * Allocates a new skew/shear matrix, specified in Renderman style, by means of a translation vector
     * tvec, a rotation vector rvec, and a rotation angle.
     * According to the Renderman specs, shearing is performed in the direction of tvec,
     * where tvec and rvec define the shearing plane. The effect is such that the skewed
     * rvec vector has the specified angle with rvec itself. (Note that the skewed rvec
     * will not have the same length as rvec; despite this, the specs talk abount "rotating"
     * rvec over an angle as specified.) The Collada skew transform for Nodes refers
     * also to the renderman specs, although their example suggests they don't understand it:
     * Rather, they claim that rvec would be the axis of rotation. 
     * The matrix argument should be a float array, with length at least 9, or it can be null, in which case
     * a Mat3f float array will be allocated. It is filled with the resulting skewing matrix, and returned
     */
    public static final float[] getSkewMatrix(float[] matrix, float angle, float[] rvec, float[] tvec) {
       // notation:
       // The shearing plane is spanned by two normalized orthogonal vectors, e0 and e1
       // e1 = normalized tvec, e0 is obtained from rvec, by subtracting any e1 component.
       // rvec = (rv0, rv1) = (rvec . e0, rvec . e1) (i.e. dot products, for projection onto axes)
       // rrot = (rr0, rr1) = rvec rotated over alpa degrees, from e0 towards e1
       // rskew = (rs0, rs1) = skewed rvec = (f*rr0, f*rr1) = (rv0, f*rr1), so f = rv0/rv1
       // skew at distance rv0 is lambda = rs1-rv1 = (rv0/rr0)*rr1 - rv1
       // So, skew at distance x1 is (x1/rv0)*lambda = x1 * d, where d = (rr1/rr0 - rv1/rv0)
       // The skew matrix maps vector v to v + d * (v . e0) e1,
       // so S = Id + d (e1e0) where (e1e0) = outer product of e1 and e0: (e1 e0)_(i,j) = e1_i * e0_j
       
       // Calculate e0 and e1:
       float[] e0 = new float[3];
       float[] e1 = new float[3];
       Vec3f.normalize(e1, tvec);
       float rv1 = Vec3f.dot(rvec, e1);
       Vec3f.set(e0, rvec);
       Vec3f.scaleAdd(e0, -rv1, e1);
       float rv0 = Vec3f.dot(rvec, e0);
       float cosa = (float) Math.cos(angle * Math.PI/180.0);
       float sina = (float) Math.sin(angle * Math.PI/180.0);
       float rr0 = rv0 * cosa - rv1 * sina;
       float rr1 = rv0 * sina + rv1 * cosa;
       
       if (rr0 < 0.000001f) throw new IllegalArgumentException("Mat4f.getSkewMatrix: illegal angle (" + angle + ")");
       
       float d = (rr1/rr0) - (rv1/rv0);
       if (matrix == null) matrix = new float[9];
       set(matrix,
        d*e1[0]*e0[0]+1.0f, d*e1[0]*e0[1],      d*e1[0]*e0[2], 
        d*e1[1]*e0[0],      d*e1[1]*e0[1]+1.0f, d*e1[1]*e0[2],
        d*e1[2]*e0[0],      d*e1[2]*e0[1],      d*e1[2]*e0[2]+1.0f
       );
       return matrix; 
    }
    
    
    
    /**
     * Transforms a Vec3 vector dest in place
     */
    public static void transformVec3f(float[]m, float[] dest) {
       transform(m, dest, dest);
    }
    
    /**
     * Transforms a Vec3 vector dest in place
     */
    public static void transformVec3f(float[]m, float[] dst, int destIndex) {
        float vx = m[m00]*dst[destIndex]+m[m01]*dst[destIndex+1]+m[m02]*dst[destIndex+2];
        float vy = m[m10]*dst[destIndex]+m[m11]*dst[destIndex+1]+m[m12]*dst[destIndex+2];
        float vz = m[m20]*dst[destIndex]+m[m21]*dst[destIndex+1]+m[m22]*dst[destIndex+2];
        dst[destIndex]  =vx;
        dst[destIndex+1]=vy;
        dst[destIndex+2]=vz;
    }
    
    
    /**
     * Transforms a Vec3 vector dest in place
     */
    public static void transformVec3f(float[]m, int mIndex, float[] dest, int dIndex) 
    {
       transform(m, mIndex, dest, dIndex, dest, dIndex);
    }
    
    public static String toString(float[]m ) {
        return "[" + m[m00] + ", " + m[m01] + ", " + m[m02] +
                "\n " + m[m10] + ", " + m[m11] + ", " + m[m12] +
                "\n " + m[m20] + ", " + m[m21] + ", " + m[m22] + "]";
    }
    
    public static String toString(float[]m ,int mIndex) {
        return "[" + m[mIndex+m00] + ", " + m[mIndex+m01] + ", " + m[mIndex+m02] +
                "\n " + m[mIndex+m10] + ", " + m[mIndex+m11] + ", " + m[mIndex+m12] +
                "\n " + m[mIndex+m20] + ", " + m[mIndex+m21] + ", " + m[mIndex+m22] + "]";
    }
    
    /**
     * The 3X3 identity matrix.
     */
    public static final float[] ID = new float[] {1f, 0f, 0f,
                                                  0f, 1f, 0f,
                                                  0f, 0f, 1f};  
    
  
}
