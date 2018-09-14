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
 * A collection of static methods for quaternions, represented by
 * float arrays of length four. Note that quaternions are also
 * Vec4f elements, so many methods from Vec4f can be used for quaternions too.
 */
public class Quat4f {

    /*
     * Some Vec4 operations are included here again, for convenience, other Vec4 operations
     * can be useful for quaternions too:
     * set, add, sub, equals, epsilonEquals, especially the variations for arrays with offsets.
     */
    
    /**
     * Offset values for quaternion s, x, y, and z components
     * q = (s, (x,y,z)), where s is the scalar part, x, y, z are the imaginary parts.
     */
    public static final int S = 0;
    public static final int X = 1;
    public static final int Y = 2;
    public static final int Z = 3;
    
    public static final int s = S;
    public static final int x = X;
    public static final int y = Y;
    public static final int z = Z;
    
    public final static double EPS2 = 1.0e-30;
    public final static double EPS = 0.000001;
    final static double SLERPEPSILON = 0.001;
    
    
    /**
     * Returns a new float[4] array with zero components
     * Note that this is NOT the identity. 
     */
    public static final float[] getQuat4f() {
       return new float[4];
    }
  
    /**
     * Returns a new float[4] array with specified components
     * No check is made that this is a unit quaternion 
     */
    public static final float[] getQuat4f(float s, float x, float y, float z) {
       return new float[] {s, x, y, z};
    }
  
    /**
     * Returns a new float[4] array initialized
     * to the identity Quat4f values: (1, 0, 0, 0)
     */
    public static final float[] getIdentity() {
       return new float[] {1.0f, 0.0f, 0.0f, 0.0f};
    }
  
    /**
     * Returns a new float[4] array with initialized components
     */
    public static final float[] getQuat4f(float[] q) {
       return new float[] {q[0], q[1], q[2], q[3]};
    }
  
    
    /**
     * Adds quaternions a to quaternion dst. 
     */
    public static void add(float[] dst, float[] a) {
       Vec4f.add(dst, a);
    }
    
    /**
     * Subtracts quaternion a from quaternion dst. 
     */
    public static final void sub(float[] dst, float[] a) {
       Vec4f.sub(dst, a);
    }
    
    /**
     * Tests for equality of two quaternions
     */
    public static final boolean equals(float[] a, float[] b){
       return Vec4f.equals(a,b); 
    }
    
    /**
     * Tests for equality of two quaternions
     */
    public static final boolean equals(float[] a, int aIndex, float[] b, int bIndex){
        return Vec4f.equals(a, aIndex, b, bIndex);
    }
    
    /**
     * Tests for equality of quaternion components within epsilon.
     */
    public static final boolean epsilonEquals(float[] a, float[] b, float epsilon){
       return Vec4f.epsilonEquals(a, b, epsilon);
    }
    
    
    /**
     * Tests for equality of quaternion components within epsilon.
     */
    public static final boolean epsilonEquals(float[] a, int aIndex, float[] b, int bIndex, float epsilon){
       return Vec4f.epsilonEquals(a, aIndex, b, bIndex, epsilon);
    }
    
    /**
     * Convert Euler angles to quaternion coefficients.
     * Which angles, which order, which signs of the angles?
     */
    public static final void setFromEulerAngles(float[] q, float heading, float attitude, float bank) {
        // Assuming the angles are in radians.
        double c1 = Math.cos(heading);
        double s1 = Math.sin(heading);
        double c2 = Math.cos(attitude);
        double s2 = Math.sin(attitude);
        double c3 = Math.cos(bank);
        double s3 = Math.sin(bank);

        double w4 = Math.sqrt(1.0 + c1 * c2 + c1 * c3 - s1 * s2 * s3 + c2 * c3) / 2.0;
        q[0] = (float) w4;
        w4 = 1/(w4 *4);
        q[1] = (float) ((c2 * s3 + c1 * s3 + s1 * s2 * c3) * w4);
        q[2] = (float) ((s1 * c2 + s1 * c3 + c1 * s2 * s3) * w4);
        q[3] = (float) ((-s1 * s3 + c1 * s2 * c3 + s2) * w4);
    }


   /**
    * calculates a quaternion representation from
    *  "roll-pitch-yaw" angles .
    * This is a rotation of the form Ry(yaw) o Rx(pitch) o Rz(roll).
    * So, the roll is around the Z axis, pitch around the X axis,
    * and yaw around the Y axis. Informally, roll is in the objects own coordinate system,
    * pitch is the angle between the objects own axis and the X-Z plane, and yaw is the 
    * "heading", obtained by rotating around the Y-axis.
    * The result (n float precision) is returned in the quaternion array q.
    */
   public  static final void setFromRollPitchYaw(float[] q, float roll, float pitch, float yaw) {
      double cx = Math.cos(pitch/2.0);
      double cy = Math.cos(yaw/2.0);
      double cz = Math.cos(roll/2.0);
      double sx = Math.sin(pitch/2.0);
      double sy = Math.sin(yaw/2.0);
      double sz = Math.sin(roll/2.0);
      q[s] = (float) (cx * cy * cz + sx * sy * sz);
      q[x] = (float) (cx * sy * sz + sx * cy * cz);
      q[y] = (float) (cx * sy * cz - sx * cy * sz);
      q[z] = (float) (cx * cy * sz - sx * sy * cz);
   }

   /**
    * calculates a quaternion representation from
    *  "roll-pitch-yaw" angles .
    * This is a rotation of the form Ry(yaw) o Rx(pitch) o Rz(roll).
    * So, the roll is around the Z axis, pitch around the X axis,
    * and yaw around the Y axis. Informally, roll is in the objects own coordinate system,
    * pitch is the angle between the objects own axis and the X-Z plane, and yaw is the 
    * "heading", obtained by rotating around the Y-axis.
    * The result (n float precision) is returned in the quaternion array q.
    * Angles are specifi3d in degrees!
    */
   public  static final void setFromRollPitchYawDegrees(float[] q, float roll, float pitch, float yaw) {
      double rh = Mat4f.degToRadf * roll/2.0f;
      double ph = Mat4f.degToRadf * pitch/2.0f;
      double yh = Mat4f.degToRadf * yaw/2.0f;

      double cx = Math.cos(ph);
      double cy = Math.cos(yh);
      double cz = Math.cos(rh);
      double sx = Math.sin(ph);
      double sy = Math.sin(yh);
      double sz = Math.sin(rh);
      q[s] = (float) (cx * cy * cz + sx * sy * sz);
      q[x] = (float) (cx * sy * sz + sx * cy * cz);
      q[y] = (float) (cx * sy * cz - sx * cy * sz);
      q[z] = (float) (cx * cy * sz - sx * sy * cz);
   }

    /**
     * Copies quaternion src to vector dst
     */
    public static final void set(float[] dst, float[] src) {
       dst[0] = src[0];
       dst[1] = src[1];
       dst[2] = src[2];
       dst[3] = src[3];
    }

    /**
     * Copies quaternion src to vector dst
     */
    public static final void set(float[] dst,int dIndex, float[] src, int sIndex) {
       dst[dIndex]   = src[sIndex];
       dst[dIndex+1] = src[sIndex+1];
       dst[dIndex+2] = src[sIndex+2];
       dst[dIndex+3] = src[sIndex+3];
    }
    
    /**
     * Sets quaternion components to specified float values.
     */
    public static final void set(float[] dst, float qs, float qx, float qy, float qz){
       dst[s] = qs;
       dst[x] = qx;
       dst[y] = qy;
       dst[z] = qz;
    }

    /**
     * Sets quaternion components to (1.0, 0.0, 0.0, 0.0)
     * This is a unit quaternion, representing the identity transform.
     */
    public static final void setIdentity(float[] dst){
       dst[s] = 1f;
       dst[x] = 0f;
       dst[y] = 0f;
       dst[z] = 0f;
    }
    
    /**
     * Sets quaternion components to (1.0, 0.0, 0.0, 0.0)
     * This is a unit quaternion, representing the identity transform.
     */
    public static final void setIdentity(float[] dst, int qIndex)
    {
       dst[qIndex+s] = 1f;
       dst[qIndex+x] = 0f;
       dst[qIndex+y] = 0f;
       dst[qIndex+z] = 0f;
    }
    
    /**
     * Sets quaternion components to (1.0, 0.0, 0.0, 0.0)
     * This is a unit quaternion, representing the identity transform.
     */
    public static final boolean isIdentity(float[] dst){
       return dst[s] == 1f && dst[x] == 0f && dst[y] == 0f && dst[z] == 0f;
    }

    /**
     * Convert Euler angles to quaternion coefficients.
     */
    public static final void setFromEulerAngles(float[] q, float[] ea) {
        setFromEulerAngles(q, ea[0], ea[1], ea[2]);
    }

    

    /**
     * Sets the quaternion coefficients from a rotation axis-angle in a float[4] array. The angle in radians.
     * The axis need not have length 1. If it has length 0, q is set to the unit quaternion (1, 0, 0, 0).
     */
    public static final void setFromAxisAngleDegrees(float[] q, float[] axis, float degrees) {
      setFromAxisAngle4f(q, axis[0], axis[1], axis[2], degrees * Mat4f.degToRadf);
    }

    /**
     * Like setFromAxisAngle4f(ax, ay, az, angle), where the axis
     * is ([[[0], aa[1], aa[2]), and the angle is aa[3].
     */
    public static final void setFromAxisAngle4f(float[] q, float[] aa) {
       setFromAxisAngle4f(q, aa[0], aa[1], aa[2], aa[3]);
    }


    /**
     * Sets the quaternion coefficients from a rotation axis (ax, ay, az) and a rotation angle, in radians.
     * The axis need not have length 1. If it has length 0, q is set to the unit quaternion (1, 0, 0, 0).
     */
    public static final void setFromAxisAngle4f(float[] q, float ax, float ay, float az, float angle) {
        double mag =  Math.sqrt(ax*ax + ay*ay + az*az);
        if (mag < EPS) {
            q[s] = 1.0f;
            q[x] = 0.0f;
            q[y] = 0.0f;
            q[z] = 0.0f;
        } else {
            q[s] = (float) (Math.cos(angle / 2.0));
            float sn = (float) (Math.sin(angle / 2.0) / mag);
            q[x] = ax * sn;
            q[y] = ay * sn;
            q[z] = az * sn;            
        }
    }
     
    /**
     * Sets an array of Quat4f quadruples from a similar aray of axis-angle quadruples.
     * The conversion for each individual axis-angle is the same as affected by setFromAxisAngle4f
     *
     */
    public static final void setQuat4fArrayFromAxisAngle4fArray(float[] q, float[] aa) {
       float ax, ay, az, angle;
       for (int offset = 0; offset<q.length; offset+=4) {  
          ax = aa[offset]; ay = aa[offset+1]; az = aa[offset+2]; angle = aa[offset+3]; 
          double mag =  Math.sqrt(ax*ax + ay*ay + az*az);
          if (mag < EPS) {
              q[offset+s] = 1.0f;
              q[offset+x] = 0.0f;
              q[offset+y] = 0.0f;
              q[offset+z] = 0.0f;
          } else {
              q[offset+s] = (float) (Math.cos(angle / 2.0));
              float sn = (float) (Math.sin(angle / 2.0) / mag);
              q[offset+x] = ax * sn;
              q[offset+y] = ay * sn;
              q[offset+z] = az * sn;            
          }
       }
    } 
     
     /**
     * Sets the quaternion coefficients from a rotation axis (ax, ay, az) and a rotation angle.
     * The axis need not have length 1. If it has length 0, q is set to the unit quaternion (1, 0, 0, 0).
     */
    public static final void setFromAxisAngle4f(float[] q, int qIndex, float[] aa, int aaIndex) {
       setFromAxisAngle4f(q, qIndex, aa[aaIndex], aa[aaIndex+1], aa[aaIndex+2], aa[aaIndex+3]);
    } 
     
    /**
     * Sets the quaternion coefficients from a rotation axis (ax, ay, az) and a rotation angle.
     * The axis need not have length 1. If it has length 0, q is set to the unit quaternion (1, 0, 0, 0).
     */
    public static final void setFromAxisAngle4f(float[] q, int qIndex, float ax, float ay, float az, float angle) {
        double mag =  Math.sqrt(ax*ax + ay*ay + az*az);
        if (mag < EPS) {
            q[qIndex+s] = 1.0f;
            q[qIndex+x] = 0.0f;
            q[qIndex+y] = 0.0f;
            q[qIndex+z] = 0.0f;
        } else {
            q[qIndex+s] = (float) (Math.cos(angle / 2.0));
            float sn = (float) (Math.sin(angle / 2.0) / mag);
            q[qIndex+x] = ax * sn;
            q[qIndex+y] = ay * sn;
            q[qIndex+z] = az * sn;            
        }
    }
    
   

    
    /**
     * Encodes a Quat4f value into AxisAngle format
     */
    public static final void setAxisAngle4fFromQuat4f(float[] aa, float[] q) {
        aa[3] = (float) (2.0 * Math.acos(q[s]));
        double len = q[x]*q[x] + q[y]*q[y] + q[z]*q[z];
        if (len > EPS) {
         len = (float) (1.0 / Math.sqrt(len));
         aa[0] = (float)(q[x]*len);
         aa[1] = (float)(q[y]*len);
         aa[2] = (float)(q[z]*len);         
        } else {
         aa[0] = 1.0f;
         aa[1] = 0.0f;
         aa[2] = 1.0f;
        }   
    }
    
    /**
     * Multiplies two quaternions, and puts the result in c:
     * c = a * b
     */
    public static final void mul(float[] c, int ci, float[] a, int ai, float[] b, int bi) {
        c[ci+s] = a[ai+s]*b[bi+s] - a[ai+x]*b[bi+x] - a[ai+y]*b[bi+y] - a[ai+z]*b[bi+z];
        c[ci+x] = a[ai+s]*b[bi+x] + b[bi+s]*a[ai+x] + a[ai+y]*b[bi+z] - a[ai+z]*b[bi+y];
        c[ci+y] = a[ai+s]*b[bi+y] + b[bi+s]*a[ai+y] + a[ai+z]*b[bi+x] - a[ai+x]*b[bi+z];
        c[ci+z] = a[ai+s]*b[bi+z] + b[bi+s]*a[ai+z] + a[ai+x]*b[bi+y] - a[ai+y]*b[bi+x];       
    }

    
    /**
     * Multiplies two quaternions, and puts the result in c:
     * c = a * b
     */
    public static final void mul(float[] c, float[] a, float[] b) {
      
        float cs = a[s]*b[s] - a[x]*b[x] - a[y]*b[y] - a[z]*b[z];
        float cx = a[s]*b[x] + b[s]*a[x] + a[y]*b[z] - a[z]*b[y];
        float cy = a[s]*b[y] + b[s]*a[y] + a[z]*b[x] - a[x]*b[z];
        c[z] = a[s]*b[z] + b[s]*a[z] + a[x]*b[y] - a[y]*b[x];     
        c[s] = cs; c[x] = cx; c[y] = cy;  
    }
    
    
    /**
     * Multiplies two quaternions, and puts the result back in a:
     * a = a * b 
     */
    public static final void mul(float[] a, float[] b) {
        mul(a, a, b);
    }
    
    /**
     * replaces quaternion a by its conjugate.
     * (x, y, z components negated, scalar component s unchanged)
     */
    public static final void conjugate(float[] a, int aIndex) {
        a[aIndex+x] = -a[aIndex+x];
        a[aIndex+y] = -a[aIndex+y];
        a[aIndex+z] = -a[aIndex+z];
    }
    
    /**
     * replaces quaternion a by its conjugate.
     * (x, y, z components negated, s component unchanged)
     */
    public static final void conjugate(float[] a) {
        a[x] = -a[x];
        a[y] = -a[y];
        a[z] = -a[z];
    }
    
    
    
    
    /**
     * replaces quaternion a by the conjugate of quaternion b
     */
    public static final void conjugate(float[] a, float[] b) {
        a[s] = b[s];
        a[x] = -b[x];
        a[y] = -b[y];
        a[z] = -b[z];
    }
    
    /**
     * replaces quaternion a by the conjugate of quaternion b
     */
    public static final void conjugate(float[] a, int aIndex, float[] b, int bIndex) {
        a[s+aIndex] = b[s+bIndex];
        a[x+aIndex] = -b[x+bIndex];
        a[y+aIndex] = -b[y+bIndex];
        a[z+aIndex] = -b[z+bIndex];
    }
    
    /**
     * replaces quaternion a by its inverse.
     * It is not assumed that the quaternion is normalized,
     * i.e. it need not have length 1.
     */
    public static final void inverse(float[] a) {
        float norm = 1.0f/(a[s]*a[s] + a[x]*a[x] + a[y]*a[y] + a[z]*a[z]);
        a[s] = norm*a[s];
        a[x] = -norm*a[x];
        a[y] = -norm*a[y];
        a[z] = -norm*a[z];
    }
    
    
    /**
     * replaces quaternion a by the inverse of quaternion b
     * It is not assumed that b is normalized,
     * i.e. it need not have length 1.
     */
    public static final void inverse(float[] a, float[] b) {
        float norm = 1.0f/(b[s]*b[s] + b[x]*b[x] + b[y]*b[y] + b[z]*b[z]);
        a[s] =  norm*b[s];
        a[x] = -norm*b[x];
        a[y] = -norm*b[y];
        a[z] = -norm*b[z];
    }
    
   /**
    * returns the square of the quaternion length
    */
   public static final float lengthSq(float[] a) {
       return Vec4f.lengthSq(a);
   }

   
   /**
    * returns the quaternion length
    */
   public static final float length(float[] a) {
       return Vec4f.length(a); 
   }
    
    /**
     * Normalizes the value of quaternion a.
     */
    public static final void normalize(float[] a, int aIndex) {
       float norm = a[aIndex+s]*a[aIndex+s] + a[aIndex+x]*a[aIndex+x] + a[aIndex+y]*a[aIndex+y] + a[aIndex+z]*a[aIndex+z];
       if (norm > 0.0f) {
          norm = 1.0f / (float)Math.sqrt(norm);
          a[aIndex+s] *= norm;
          a[aIndex+x] *= norm;
          a[aIndex+y] *= norm;
          a[aIndex+z] *= norm;
       }
    }
    
    /**
     * Normalizes the value of quaternion a.
     */
    public static final void normalize(float[] a) {
       float norm = a[s]*a[s] + a[x]*a[x] + a[y]*a[y] + a[z]*a[z];
       if (norm > 0.0f) {
          norm = 1.0f / (float)Math.sqrt(norm);
          a[s] *= norm;
          a[x] *= norm;
          a[y] *= norm;
          a[z] *= norm;
       } 
    }

    /**
     * Sets quaternion a to the normalized version of quaternion b.
     */
    public static final void normalize(float[] a, float[] b) {
       float norm = b[s]*b[s] + b[x]*b[x] + b[y]*b[y] + b[z]*b[z];

       if (norm > 0.0f) {
          norm = 1.0f / (float)Math.sqrt(norm);
          a[s] = norm*b[s];
          a[x] = norm*b[x];
          a[y] = norm*b[y];
          a[z] = norm*b[z];
       } else {
          a[s] = 0.0f;
          a[x] = 0.0f;
          a[y] = 0.0f;
          a[z] = 0.0f;
       }
    }
    
    
   
    
    
    /**
     * Performs a great circle interpolation (slerp) between two quaternions
     * q1 and q2, and places the result in quaternion qr.
     */
    public static final void interpolate(float[] qr, float[] q1, float[] q2, float alpha) {
        qr[s] = q1[s];
        qr[x] = q1[x];
        qr[y] = q1[y];
        qr[z] = q1[z];
        interpolate(qr, q2, alpha);
    }
    
    
    
    
    
    /**
     * Performs a great circle interpolation (slerp) between two quaternions
     * qr and q, and places the result back into quaternion qr.
     */
    public static final void interpolate(float[] qr, float[] q, float alpha) {
        double cosOmega = qr[s]*q[s] + qr[x]*q[x] + qr[y]*q[y] + qr[z]*q[z];
        
        if (cosOmega < 0) {
            qr[s] = -qr[s];
            qr[x] = -qr[x];
            qr[y] = -qr[y];
            qr[z] = -qr[z];
            cosOmega = -cosOmega;            
        } 
        float s1, s2;
        if (1-cosOmega < SLERPEPSILON) { // go for linear interpolation, rather than slerp interpolation
            s1 = 1.0f-alpha;
            s2 = alpha;
        } else { // slerp interpolation 
            double omega = Math.acos(cosOmega);         
            double sinOmega = Math.sin(omega);
            s1 = (float) (Math.sin((1.0-alpha)*omega)/sinOmega);
            s2 = (float) (Math.sin( alpha*omega)/sinOmega);
        } 
        qr[s] = s1*qr[s] + s2*q[s];
        qr[x] = s1*qr[x] + s2*q[x];
        qr[y] = s1*qr[y] + s2*q[y];
        qr[z] = s1*qr[z] + s2*q[z];
   }
    
    
   /**
    * Performs a great circle interpolation (slerp) between  quaternions taken from arrays
    * q1 and q2, and places the results in array qr. The arrays are assumed to have equals size,
    * which should be a mutiple of 4. Each consecutive four floats are considered to be one quaternion
    * in the standard order: (s, x, y, z).
    */
   public static final void interpolate(float[] qr, int qrIndex, float[] q1, int q1Index, float[] q2, int q2Index, float alpha) {
      float q1s, q1x, q1y, q1z;
      float q2s, q2x, q2y, q2z;
      
      q1s = q1[q1Index+s];
      q1x = q1[q1Index+x];
      q1y = q1[q1Index+y];
      q1z = q1[q1Index+z];
      
      q2s = q2[q2Index+s];
      q2x = q2[q2Index+x];
      q2y = q2[q2Index+y];
      q2z = q2[q2Index+z];
         
      double cosOmega = q1s*q2s + q1x*q2x + q1y*q2y + q1z*q2z;
      if (cosOmega < 0) {
          q1s = -q1s;
          q1x = -q1x;
          q1y = -q1y;
          q1z = -q1z;
          cosOmega = -cosOmega;            
      } 
      float s1, s2;
      if (1-cosOmega < SLERPEPSILON) { // go for linear interpolation, rather than slerp interpolation
          s1 = 1.0f-alpha;
          s2 = alpha;
      } else { // slerp interpolation 
          double omega = Math.acos(cosOmega);         
          double sinOmega = Math.sin(omega);
          s1 = (float) (Math.sin((1.0-alpha)*omega)/sinOmega);
          s2 = (float) (Math.sin( alpha*omega)/sinOmega);
      } 
      qr[qrIndex+s] = s1*q1s + s2*q2s;
      qr[qrIndex+x] = s1*q1x + s2*q2x;
      qr[qrIndex+y] = s1*q1y + s2*q2y;
      qr[qrIndex+z] = s1*q1z + s2*q2z;
  
   }  
    
    
   /**
    * Performs a great circle interpolation (slerp) between  quaternions taken from arrays
    * q1 and q2, and places the results in array qr. The arrays are assumed to have equals size,
    * which should be a mutiple of 4. Each consecutive four floats are considered to be one quaternion
    * in the standard order: (s, x, y, z).
    */
   public static final void interpolateArrays(float[] qr, float[] q1, float[] q2, float alpha) {
      for (int offset=0; offset<qr.length; offset+=4) {
         interpolate(qr, offset, q1, offset, q2, offset, alpha);

      }

   } 
   
   /**
    * rotates a vector with a quaternion, assumes the quaternion is length 1 
    * transforms v, and also returns it.
    */
   public static final float[] transformVec3f(float[] q, int qIndex, float[] v, int vIndex) {
     // calculate (qs, qx, qy, qz) = q *(0,v)
     float qs = - q[qIndex+X]*v[vIndex] - q[qIndex+Y]*v[vIndex+1] - q[qIndex+Z]*v[vIndex+2];
     float qx =   q[qIndex+S]*v[vIndex] + q[qIndex+Y]*v[vIndex+2] - q[qIndex+Z]*v[vIndex+1];
     float qy =   q[qIndex+S]*v[vIndex+1] + q[qIndex+Z]*v[vIndex] - q[qIndex+X]*v[vIndex+2];
     float qz =   q[qIndex+S]*v[vIndex+2] + q[qIndex+X]*v[vIndex+1] - q[qIndex+Y]*v[vIndex];   
       
     // Calculate v = imaginary part of (q *(0,v)) * conj(q)
     v[vIndex]   = q[qIndex+S]*qx - qs*q[qIndex+X] - qy * q[qIndex+Z] + qz*q[qIndex+Y] ;
     v[vIndex+1] = q[qIndex+S]*qy - qs*q[qIndex+Y] - qz * q[qIndex+X] + qx*q[qIndex+Z];
     v[vIndex+2] = q[qIndex+S]*qz - qs*q[qIndex+Z] - qx * q[qIndex+Y] + qy*q[qIndex+X];  
     return v;
   }
   
    /**
     * rotates a vector with a quaternion, assumes the quaternion is length 1 
     * transforms v, and also returns it.
     */
    public static final float[] transformVec3f(float[] q, float[] v) {
      // calculate (qs, qx, qy, qz) = q *(0,v)
      float qs = - q[X]*v[0] - q[Y]*v[1] - q[Z]*v[2];
      float qx = q[S]*v[0] + q[Y]*v[2] - q[Z]*v[1];
      float qy = q[S]*v[1] + q[Z]*v[0] - q[X]*v[2];
      float qz = q[S]*v[2] + q[X]*v[1] - q[Y]*v[0];   
        
      // Calculate v = imaginary part of (q *(0,v)) * conj(q)
      v[0] = q[S]*qx - qs*q[X] - qy * q[Z] + qz*q[Y] ;
      v[1] = q[S]*qy - qs*q[Y] - qz * q[X] + qx*q[Z];
      v[2] = q[S]*qz - qs*q[Z] - qx * q[Y] + qy*q[X];  
      return v;
    }
    
    public static String toString(float[]a, int aIndex) {
        return "(" + a[aIndex] + ", " + a[aIndex+1] + "," + a[aIndex+2] + "," + a[aIndex+3] + ")";
       
    }
   
   
   
   public static String toString(float[]a ) {
        return "(" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + ")";
       
    }
    
    
    /**
    * Calculates qout=qin^p
    * qout can be the same array as qin
    */
   public static final void pow(float []qin, float p, float[] qout)
   {
      float angle = (float)Math.acos(qin[S]);
        float sinAngle = (float)Math.sin(angle);        
        
        float vx,vy,vz;
        if(sinAngle>EPS)
        {
           vx = qin[X]/sinAngle;
           vy = qin[Y]/sinAngle;
           vz = qin[Z]/sinAngle;
           
           angle *= p;
        
           sinAngle = (float)Math.sin(angle);
           qout[X] = vx * sinAngle;
           qout[Y] = vy * sinAngle;
           qout[Z] = vz * sinAngle;
           qout[S] = (float)Math.cos(angle);
        }
        else
        {
         qout[X]=qin[X];
         qout[Y]=qin[Y];
         qout[Z]=qin[Z];
         qout[S]=qin[S];
        }
   }
   
   
   /**
    * qin=qin^p 
    */
   public static final void pow(float []qin, float p)
   {
      pow(qin,p,qin);
   }
   
   /**
    * Calculates the instantatious angular velocity from the quaternion rate and quaternion rotation
    * w = 2*qrate*q^-1
    */
   public static final void setAngularVelocityFromQuat4f(float []avel, float q[], float qrate[])
   {
       avel[0] = -qrate[s]*q[x] + q[s]*qrate[x] - qrate[y]*q[z] + qrate[z]*q[y];
       avel[1] = -qrate[s]*q[y] + q[s]*qrate[y] - qrate[z]*q[x] + qrate[x]*q[z];
       avel[2] = -qrate[s]*q[z] + q[s]*qrate[z] - qrate[x]*q[y] + qrate[y]*q[x];  
       Vec3f.scale(2,avel);
   }
   
   /**
    * Calculates the instantatious angular velocity from the quaternion rate and quaternion rotation
    * w = 2*qrate*q^-1
    */
   public static final void setAngularVelocityFromQuat4f(float []avel, int aVelIndex, float q[], int qIndex, float qrate[], int qRateIndex)
   {
       avel[aVelIndex]   = -qrate[qRateIndex+s]*q[qIndex+x] + q[qIndex+s]*qrate[qRateIndex+x] - qrate[qRateIndex+y]*q[qIndex+z] + qrate[qRateIndex+z]*q[qIndex+y];
       avel[aVelIndex+1] = -qrate[qRateIndex+s]*q[qIndex+y] + q[qIndex+s]*qrate[qRateIndex+y] - qrate[qRateIndex+z]*q[qIndex+x] + qrate[qRateIndex+x]*q[qIndex+z];
       avel[aVelIndex+2] = -qrate[qRateIndex+s]*q[qIndex+z] + q[qIndex+s]*qrate[qRateIndex+z] - qrate[qRateIndex+x]*q[qIndex+y] + qrate[qRateIndex+y]*q[qIndex+x];  
       Vec3f.scale(2,avel,aVelIndex);
   }
   
   /**
    * Calculates the instantatious angular acceleration from the quaternion, quaternion rate and quaternion rate diff
    * [s w']^T = 2 * qrate' * q^-1
    */
   public static final void setAngularAccelerationFromQuat4f(float []aacc,float q[], float qratediff[])
   {
       //qrate' * q^-1 
       aacc[0] = -qratediff[s]*q[x] + q[s]*qratediff[x] - qratediff[y]*q[z] + qratediff[z]*q[y];
       aacc[1] = -qratediff[s]*q[y] + q[s]*qratediff[y] - qratediff[z]*q[x] + qratediff[x]*q[z];
       aacc[2] = -qratediff[s]*q[z] + q[s]*qratediff[z] - qratediff[x]*q[y] + qratediff[y]*q[x];  
       
       Vec3f.scale(2,aacc);
   }
   
   /**
    * Calculates the instantatious angular acceleration from the quaternion, quaternion rate and quaternion rate diff
    * a = w' 
    *   = 2*qrate'*(q^-1)
    */
   public static final void setAngularAccelerationFromQuat4f(float []aacc, int aaccIndex, float q[], int qIndex, float qratediff[], int qRateDiffIndex)
   {
       //qrate' * q^-1 
       aacc[aaccIndex]   = -qratediff[qRateDiffIndex+s]*q[qIndex+x] + q[qIndex+s]*qratediff[qRateDiffIndex+x] - qratediff[qRateDiffIndex+y]*q[qIndex+z] + qratediff[qRateDiffIndex+z]*q[qIndex+y];
       aacc[aaccIndex+1] = -qratediff[qRateDiffIndex+s]*q[qIndex+y] + q[qIndex+s]*qratediff[qRateDiffIndex+y] - qratediff[qRateDiffIndex+z]*q[qIndex+x] + qratediff[qRateDiffIndex+x]*q[qIndex+z];
       aacc[aaccIndex+2] = -qratediff[qRateDiffIndex+s]*q[qIndex+z] + q[qIndex+s]*qratediff[qRateDiffIndex+z] - qratediff[qRateDiffIndex+x]*q[qIndex+y] + qratediff[qRateDiffIndex+y]*q[qIndex+x];  
       
       Vec3f.scale(2,aacc,aaccIndex);
   }
   
   
   /**
     * Quats close to either (1.0, 0.0, 0.0, 0.0) or (-1.0, 0.0, 0.0, 0.0)
     * are rounded towards these values, provided the first element deviates less
     * from 1 or -1 by an amount eps
     */
    public static void smooth(float[] q, float eps) {
       if (Math.abs(q[0]-1.0f) < eps) {
           q[0] = 1.0f; q[1] = 0.0f;  q[2] = 0.0f; q[3] = 0.0f;
       } else if (Math.abs(q[0]+1.0f) < eps) {
           q[0] = -1.0f; q[1] = 0.0f;  q[2] = 0.0f; q[3] = 0.0f;
       }     
   
    }
   
   
}