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
 * Efficient implementation of the 6x6 spatial transform matrix
 * As defined in
 * 
 * Rigid Body Dynamics Algorithms
 * Roy Featherstone
 * 2007
 * 
 * using a 12-element float array
 * @author welberge
 */
public class SpatialTransform
{
    public static int ROFFSET = 9;    
    
    
    /**
     * Sets the spatial transform
     * @param dst the spatial transform (lenght 12 float array)
     * @param m 3x3 rotation matrix, specified as length 9 float array
     * @param tr 3x1 translation vector
     */
    public static void setFromMat3fVec3f(float []dst, float m[], float tr[])
    {
        for(int i=0;i<9;i++)
        {
            dst[i] = m[i];            
        }
        dst[ROFFSET] = tr[0];
        dst[ROFFSET+1] = tr[1];
        dst[ROFFSET+2] = tr[2];
    }
    
    /**
     * Sets the spatial transform
     * @param dst the spatial transform (lenght 12 float array)
     * @param q quaternion
     * @param tr 3x1 translation vector
     */
    public static void setFromQuat4fVec3f(float []dst, float q[], float tr[])
    {
        Mat3f.setFromQuatScale(dst, q, 1.0f);
        dst[ROFFSET] = tr[0];
        dst[ROFFSET+1] = tr[1];
        dst[ROFFSET+2] = tr[2];
    }
    
    /**
     * Sets the spatial transform from another spatial transform
     */
    public static void set(float []dst, float src[])
    {
        System.arraycopy(src, 0, dst, 0, 12);
    }
    
    /**
     * Sets the spatial transform from another spatial transform
     */
    public static void set(float []dst, int dIndex, float src[], int srcIndex)
    {
        System.arraycopy(src, srcIndex, dst, dIndex, 12);
    }
    
    /**
     * Sets the spatial transform
     * @param dst the spatial transform (lenght 12 float array)
     * @param q quaternion
     * @param tr 3x1 translation vector
     */
    public static void setFromQuat4fVec3f(float []dst, int dstIndex, float q[], int qIndex, float tr[], int trIndex)
    {
        Mat3f.setFromQuatScale(dst,dstIndex, q,qIndex, 1.0f);
        //Mat3f.transpose(dst,dstIndex);
        dst[dstIndex+ROFFSET] = tr[trIndex];
        dst[dstIndex+ROFFSET+1] = tr[trIndex+1];
        dst[dstIndex+ROFFSET+2] = tr[trIndex+2];
    }
    
    /**
     * Transforms a vec6
     * @param trans the spatial transform
     * @param src the vec6
     * dest = spx(E,r)spv(v,v0)
     */
    public static void transformMotion(float dest[], float trans[], float src[])
    {
        //mv(E * w, E( v - r x w ))
        Mat3f.transform(trans, dest, src);
        
        //v - r x w
        float cx = src[3] - (trans[ROFFSET+1]*src[2] - trans[ROFFSET+2] * src[1]);
        float cy = src[4] - (trans[ROFFSET+2]*src[0] - trans[ROFFSET]   * src[2]);
        float cz = src[5] - (trans[ROFFSET]  *src[1] - trans[ROFFSET+1] * src[0]);
        
        Mat3f.transform(trans, dest, 3, cx,cy,cz);
    }
    
    /**
     * Transforms a spatial motion vector
     * @param trans the spatial transform
     * @param src the vec6
     * dest = spx(E,r)spv(v,v0)
     */
    public static void transformMotion(float dest[], int destI, float trans[], int transI, float src[], int srcI)
    {
        //spv(E w, E( v - r x w))
        Mat3f.transform(trans, transI, dest, destI, src, srcI);
        
        //v - r x w
        float cx = src[3+srcI] - (trans[transI+ROFFSET+1]*src[srcI+2] - trans[transI+ROFFSET+2] * src[srcI+1]);
        float cy = src[4+srcI] - (trans[transI+ROFFSET+2]*src[srcI] - trans[transI+ROFFSET]   * src[srcI+2]);
        float cz = src[5+srcI] - (trans[transI+ROFFSET]  *src[srcI+1] - trans[transI+ROFFSET+1] * src[srcI]);
        
        Mat3f.transform(trans, transI, dest, 3+destI, cx,cy,cz);
    }
    
    /**
     * Transforms a spatial force vector
     * Vec6 src, Vec6 dest, 12 float  spatial transform trans.
     * @param trans the spatial transform
     */
    public static void transformForce(float[] dest, float[] trans, float[] src)
    {
        // trans = (E,r)   (Mat3f E, Vec3f r)
        // src = (n, f)    (Vec3f n Vec3f f)
        //dest = (E * (n - r x f), E * f)  
        
        //save r x f in dest
        Vec3f.cross(dest, 3, trans, ROFFSET, src, 3);
        
        //n - r x f
        Vec3f.sub(dest, 0, src, 0, dest, 3);
        
        //E(n - r x f)
        Mat3f.transform(trans,0, dest, 3);
        
        //E * f
        Mat3f.transform(trans,0,dest, 3,src,3);
    }
    
    /**
     * Transforms a spatial force vector
     * @param trans the spatial transform
     */
    public static void transformForce(float dest[], int destIndex, float trans[], int transIndex, float src[], int srcIndex)
    {
        //(E * (n - r x f), E * f)
        
        //r x f
        Vec3f.cross(dest, destIndex+3, trans, transIndex+ROFFSET, src, srcIndex+3);
        
        //n - r x f
        Vec3f.sub(dest, destIndex, src, srcIndex, dest, destIndex+3);
        
        //E * (n - r x f)
        Mat3f.transform(trans,transIndex, dest, destIndex);
        
        //E * f
        Mat3f.transform(trans,transIndex,dest, destIndex+3, src,srcIndex+3);
    }
    
    /**
     * Transforms a spatial motion vector with the transpose
     * @param trans the spatial transform
     * @param src the vec6     
     */
    public static void transformMotionTranspose(float dest[], float trans[], float src[])
    {
        //mv( E^T * w, r x (E^T * w) + E^T * v)
        
        //E^T * w
        Mat3f.transformTranspose(trans, dest, src);
        
        //E^T * v
        Mat3f.transformTranspose( trans,0, dest, 3,src,3);
        
        //E^T * v + r x (E^T * w)
        dest[3] += trans[ROFFSET+1]*dest[2] - trans[ROFFSET+2] * dest[1];
        dest[4] += trans[ROFFSET+2]*dest[0] - trans[ROFFSET]   * dest[2];
        dest[5] += trans[ROFFSET]  *dest[1] - trans[ROFFSET+1] * dest[0];
    }
    
    /**
     * Transforms a spatial motion vector with the transpose
     * @param trans the spatial transform
     * @param src the vec6     
     */
    public static void transformMotionTranspose(float dest[], int destI, float trans[], int transI, float src[], int srcI)
    {
        //mv(E^T * w, r x (E^T * w)+E^T * v)
        
        //E^T * w
        Mat3f.transformTranspose( trans, transI, dest, destI,src, srcI);
        
        //E^T * v
        Mat3f.transformTranspose( trans, transI, dest, 3+destI,src,3+srcI);
        
        //E^T * v + r x (E^T * w)
        dest[destI+3]   += trans[transI+ROFFSET+1]*dest[destI+2] - trans[transI+ROFFSET+2] * dest[destI+1];
        dest[destI+4] += trans[transI+ROFFSET+2]*dest[destI]   - trans[transI+ROFFSET]   * dest[destI+2];
        dest[destI+5] += trans[transI+ROFFSET]  *dest[destI+1] - trans[transI+ROFFSET+1] * dest[destI];
    }
    
    /**
     * Transforms a spatial force vector with the transpose
     */
    public static void transformForceTranspose(float dest[], float trans[], float src[])
    {
        //fv(E^T*n + r x E^T * f, E^T * f)
        
        //E^T * f
        Mat3f.transformTranspose(trans, 0, dest, 0, src, 3);
        float fx = dest[0];
        float fy = dest[1];
        float fz = dest[2];
        
        //r x E^T * f
        Vec3f.cross(dest, 3, trans, ROFFSET, dest, 0);
        
        //E^T * n
        Mat3f.transformTranspose(trans,dest,src);
        
        //E^T * n + r x E^T * f
        Vec3f.add(dest, 0, dest, 3);
        
        //E^T * f
        Vec3f.set(dest,3,fx,fy,fz);
    }
    
    /**
     * Transforms a spatial force vector with the transpose
     */
    public static void transformForceTranspose(float dest[], int destIndex, float trans[], int transIndex, float src[], int srcIndex)
    {
        //fv(E^T*n + r x E^T * f, E^T * f)
        
        //E^T * f
        Mat3f.transformTranspose( trans, transIndex, dest, destIndex,src, srcIndex+3);
        float fx = dest[destIndex];
        float fy = dest[destIndex+1];
        float fz = dest[destIndex+2];
        
        //r x E^T * f
        Vec3f.cross( dest, destIndex+3,trans, transIndex+ROFFSET, dest, destIndex);
        
        //E^T * n
        Mat3f.transformTranspose(trans,transIndex,dest,destIndex,src,srcIndex);
        
        //E^T * n + r x E^T * f
        Vec3f.add(dest, destIndex, dest, destIndex+3);
        
        //E^T * f
        Vec3f.set(dest,destIndex+3,fx,fy,fz);
    }
    
    /**
     * Multiplies spatial transforms a and b and stores the result in dest
     * dest is not allowed to be aliased with a or b 
     */
    public static void mul(float dest[], float a[], float b[])
    {
        //spx(E1,r1)spx(E2,r2)=spx(E1E2,r2+E2^-1r1)        
        Mat3f.mul(dest, a, b);
        Mat3f.transformTranspose(b, 0, dest, ROFFSET,a, ROFFSET);
        Vec3f.add(dest, ROFFSET, b, ROFFSET);
    }
    
    /**
     * Multiplies spatial transforms a and b and stores the result in dest
     * dest is not allowed to be aliased with a or b 
     */
    public static void mul(float dest[], int dIndex, float a[], int aIndex, float b[], int bIndex)
    {
        //spx(E1,r1)spx(E2,r2)=spx(E1E2,r2+E2^-1r1)        
        Mat3f.mul(dest, dIndex, a, aIndex, b, bIndex);
        Mat3f.transformTranspose(b, bIndex, dest, dIndex+ROFFSET,a, aIndex+ROFFSET);
        Vec3f.add(dest, dIndex+ROFFSET, b, bIndex+ROFFSET);
    }
    
    /**
     * Sets the transpose of dst in dst 
     */
    public static void transpose(float dst[])
    {
        Mat3f.transform(dst, 0, dst, ROFFSET);
        Vec3f.scale(-1, dst,ROFFSET);
        Mat3f.transpose(dst);
    }  
    
    /**
     * Sets the transpose of a in dest, dest can not be aliased in a
     */
    public static void transpose(float dest[], float a[])
    {
        Mat3f.transform(a, 0, dest, ROFFSET, a, ROFFSET);
        Vec3f.scale(-1, dest,ROFFSET);
        Mat3f.transpose(dest, a);
    } 
    
    public static boolean epsilonEquals(float[] a, float[] b, float epsilon)
    {
        if (!Mat3f.epsilonEquals(a, b, epsilon))return false;
        return Vec3f.epsilonEquals(a, ROFFSET, b, ROFFSET, epsilon);
    }
    
    public static String toString(float[] a)
    {
        return Mat3f.toString(a)+Vec3f.toString(a, ROFFSET);
    }
    
    public static String toString(float[] a, int aIndex)
    {
        return Mat3f.toString(a,aIndex)+Vec3f.toString(a, aIndex+ROFFSET);
    }
    /**
     * The identity matrix.
     */
    public static final float[] ID = new float[] {1f, 0f, 0f,
                                                  0f, 1f, 0f,
                                                  0f, 0f, 1f,
                                                  0f, 0f, 0f};  
    
}
