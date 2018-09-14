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
 * A collection of methods for spatial vectors 
 * As defined in
 * 
 * Rigid Body Dynamics Algorithms
 * Roy Featherstone
 * 2007
 *  
 * Spatial vectors can be stored within arrays of length 6,
 * or they can be stored inside a larger float array a, together with an 
 * integer offset &quot;index&quot;  into that array. This represents a vector
 * with components (a[index], a[index+1], a[index+2]).
 * The methods from this class never allocate new arrays; rather, they assume that
 * results are to be stored into some existing &quot;destination&quot; array.
 */
public class SpatialVec
{
    /**
     * Copies the src 6-vector to the dst 6-vector
     */
    public static void set(float dst[], float v0, float v1, float v2, float v3, float v4, float v5)
    {
        dst[0] = v0;
        dst[1] = v1;
        dst[2] = v2;
        dst[3] = v3;
        dst[4] = v4;
        dst[5] = v5;
    }
    
    /**
     * Copies the src 6-vector to the dst 6-vector
     */
    public static void set(float dst[], float[]src)
    {
        for(int i=0;i<6;i++)
        {
            dst[i]=src[i];        
        }
    }
    
    /**
     * Copies the src 6-vector to the dst 6-vector
     */
    public static void set(float dst[], int dstIndex, float[]src, int srcIndex)
    {
        for(int i=0;i<6;i++)
        {
            dst[dstIndex+i]=src[srcIndex+i];        
        }
    }
    
    /**
     * Sets the dst 6-vector from two 3-vectors 
     * dst=[w v0]^T
     */
    public static void set(float dst[], float[]w, float[]v0)
    {
        dst[0]=w[0];
        dst[1]=w[1];
        dst[2]=w[2];
        dst[3]=v0[0];
        dst[4]=v0[1];
        dst[5]=v0[2];
    }
    
    /**
     * Set the spatial acceleration vector from 'traditional' angular velocity, velocity, angular 
     * acceleration and acceleration 
     */
    public static void setAcc(float dst[], float[] w, float[] v, float[] wDiff, float[] a)
    {
        //a - w x v
        Vec3f.cross(dst, w, v);
        Vec3f.set(dst,3,a,0);
        Vec3f.sub(dst,3,dst,0);
        
        //wDiff
        Vec3f.set(dst,wDiff);        
    }
    
    /**
     * dst = a+b
     */
    public static void add(float dst[], float a[], float b[])
    {
        dst[0]=a[0]+b[0];
        dst[1]=a[1]+b[1];
        dst[2]=a[2]+b[2];
        dst[3]=a[3]+b[3];
        dst[4]=a[4]+b[4];
        dst[5]=a[5]+b[5];
    }
    
    /**
     * dst = a-b
     */
    public static void sub(float dst[], float a[], float b[])
    {
        dst[0]=a[0]-b[0];
        dst[1]=a[1]-b[1];
        dst[2]=a[2]-b[2];
        dst[3]=a[3]-b[3];
        dst[4]=a[4]-b[4];
        dst[5]=a[5]-b[5];
    }
    
    /**
     * dst = dst+a
     */
    public static void add(float dst[], float a[])
    {
        dst[0]+=a[0];
        dst[1]+=a[1];
        dst[2]+=a[2];
        dst[3]+=a[3];
        dst[4]+=a[4];
        dst[5]+=a[5];
    }
    
    /**
     * dst = dst-a
     */
    public static void sub(float dst[], float a[])
    {
        dst[0]-=a[0];
        dst[1]-=a[1];
        dst[2]-=a[2];
        dst[3]-=a[3];
        dst[4]-=a[4];
        dst[5]-=a[5];
    }
    
    /**
     * dst = dst+a
     */
    public static void add(float dst[], int dstIndex, float a[], int aIndex)
    {
        dst[dstIndex]   += a[aIndex];
        dst[dstIndex+1] += a[aIndex+1];
        dst[dstIndex+2] += a[aIndex+2];
        dst[dstIndex+3] += a[aIndex+3];
        dst[dstIndex+4] += a[aIndex+4];
        dst[dstIndex+5] += a[aIndex+5];
    }
    
    /**
     * dst = dst-a
     */
    public static void sub(float dst[], int dstIndex, float a[], int aIndex)
    {
        dst[dstIndex]   -= a[aIndex];
        dst[dstIndex+1] -= a[aIndex+1];
        dst[dstIndex+2] -= a[aIndex+2];
        dst[dstIndex+3] -= a[aIndex+3];
        dst[dstIndex+4] -= a[aIndex+4];
        dst[dstIndex+5] -= a[aIndex+5];
    }
    
    /**
     * Spatial dot product     
     */
    public static float dot(float a[], float b[])
    {
        return Vec3f.dot(a,0, b,3)+Vec3f.dot(a,3, b,0);
    }
    
    /**
     * Spatial dot product     
     */
    public static float dot(float a[], int aIndex, float b[], int bIndex)
    {
        return Vec3f.dot(a,aIndex, b,bIndex+3)+Vec3f.dot(a,aIndex+3, b,bIndex);
    }
    
    /**
     * Cross product
     */
    public static void cross(float[]dst, float a[],float b[])
    {
        //(wa x wb, wa x vb + va x wb)
        
        //aw x bv
        Vec3f.cross(dst, 0, a, 0, b, 3);
        
        //av x bw
        Vec3f.cross(dst, 3, a, 3, b, 0);
        
        //aw x bv + av x bw
        Vec3f.add(dst,3, dst, 0);
        
        //aw x bw
        Vec3f.cross(dst, a, b);    
    }
    
    /**
     * velocity x velocity product
     */
    public static void cross(float[]dst, int dstIndex, float a[], int aIndex, float b[], int bIndex)
    {
        //(aw x bw, aw x bv + av x bw)
        
        //aw x bv
        Vec3f.cross(dst, dstIndex, a, aIndex, b, bIndex+3);
        
        //av x bw
        Vec3f.cross(dst, dstIndex+3, a, aIndex+3, b, bIndex);
        
        //aw x bv + av x bw
        Vec3f.add(dst,dstIndex+3, dst, dstIndex);
        
        //aw x bw
        Vec3f.cross(dst, dstIndex, a, aIndex, b, bIndex);        
    }
    
    /**
     * movement x force, Featherstone's x* operation
     */
    public static void crossForce(float[]dst, float v[], float f[])
    {
        //fv(vw x fn + vv x ff, vw x ff)
        Vec3f.cross(dst, v, f);
        Vec3f.cross(dst, 3, v, 3, f, 3);
        Vec3f.add(dst,0,dst,3);
        Vec3f.cross(dst,3, v, 0, f,3);
    }
    
    /**
     * movement x force, Featherstone's x* operation
     */
    public static void crossForce(float[]dst, int dstIndex, float v[], int vIndex, float f[], int fIndex)
    {
        //fv(vw x fn + vv x ff, vw x ff)
        
        //vw x fn
        Vec3f.cross(dst, dstIndex, v, vIndex, f, fIndex);
        
        //vv x ff
        Vec3f.cross(dst, dstIndex+3, v, vIndex+3, f, fIndex+3);
        
        //vw x fn + vw x ff
        Vec3f.add(dst,dstIndex,dst,dstIndex+3);
        
        //vw x ff
        Vec3f.cross(dst,dstIndex+3, v, vIndex, f,fIndex+3);
    }
    
    /**
     * Tests for equality of vector components within epsilon.
     */
    public static final boolean epsilonEquals(float[] a, float[] b, float epsilon)
    {
        return Vec3f.epsilonEquals(a, b, epsilon) && Vec3f.epsilonEquals(a, 3, b,3, epsilon); 
    }
    
    /**
     * Tests for equality of vector components within epsilon.
     */
    public static final boolean epsilonEquals(float[] a, int aIndex, float[] b, int bIndex, float epsilon)
    {
        return Vec3f.epsilonEquals(a,aIndex, b, bIndex,epsilon) && Vec3f.epsilonEquals(a,aIndex+3, b,bIndex+3, epsilon); 
    }
    
    /**
     * String representation
     */
    public static final String toString(float []a)
    {
        return Vec3f.toString(a)+Vec3f.toString(a,3);
    }
    
    /**
     * String representation
     */
    public static final String toString(float []a, int aIndex)
    {
        return Vec3f.toString(a,aIndex)+Vec3f.toString(a,aIndex+3);
    }
    
    /**
     * The zero vector
     */
    public static final float[] ZERO = new float[] {0f, 0f, 0f, 0f, 0f, 0f};
                                                  
                                
}
