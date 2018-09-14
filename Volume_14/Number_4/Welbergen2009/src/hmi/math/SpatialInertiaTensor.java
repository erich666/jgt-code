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
 * Efficient implementation of the 6x6 spatial inertia tensor 
 * As defined in
 * 
 * Rigid Body Dynamics Algorithms
 * Roy Featherstone
 * 2007
 * 
 * using a 13-element float array
 * @author welberge
 */
public class SpatialInertiaTensor
{
    public static final int h = 9;
    public static final int m = 12;
    
    /**
     * Sets the tensor, rotI is the rotational inertia tensor at the center of mass C
     * pos is the vector OC, with O the origin of the body
     */
    public static void set(float I[], float[] rotI,float pos[], float mass)
    {
        //I = rotI - m * cx * cx        
        //cx * cx
        Mat3f.skew(I, pos);
        Mat3f.mul(I, I);
        //-m * cx * cx
        Mat3f.scale(I, -mass);
        //I = rotI - m * cx * cx
        Mat3f.add(I,rotI);
        
        
        I[m]=mass;
        
        // h = m * c
        Vec3f.set(I,h,pos,0);
        Vec3f.scale(mass,I,h);
    }
    
    /**
     * Sets the tensor, rotI is the rotational inertia tensor at the center of mass C
     * pos is the vector OC, with O the origin of the body
     */
    public static void set(float I[], int iIndex, float[] rotI,float pos[], float mass)
    {
        //I = rotI - m * cx * cx        
        //cx * cx
        Mat3f.skew(I, iIndex, pos, 0);
        Mat3f.mul(I, iIndex, I, iIndex);
        //-m * cx * cx
        Mat3f.scale(I, iIndex, -mass);
        //I = rotI - m * cx * cx
        Mat3f.add(I,iIndex, rotI,0);
        
        
        I[iIndex+m]=mass;
        
        // h = m * c
        Vec3f.set(I,iIndex+h,pos,0);
        Vec3f.scale(mass,I,iIndex+h);
    }
    
    /**
     * Sets the tensor with the coordinate frame at the center of mass, that is, c = 0
     * rotI is the rotational inertia tensor at the center of mass
     */
    public static void set(float I[], float[] rotI, float mass)
    {
        Mat3f.set(I,rotI);
        Vec3f.set(I,h,0,0,0);
        I[m]=mass;
    }
    
    /**
     * vdest = I*a 
     */
    public static void transformSpatialVec(float vdest[], float I[], float a[])
    {
        //fv(I*aw + h x av, m*av - h x aw 
        
        //I * aw
        Mat3f.transform(I, vdest, a);
        
        //h x av
        Vec3f.cross(vdest, 3, I,h, a,3);
        
        //I * aw + h x av
        Vec3f.add(vdest,0,vdest,3);
        
        //h x aw
        Vec3f.cross(vdest,3,I,h,a,0);
        
        //-h x aw
        Vec3f.scale(-1f, vdest,3);
        
        //-h x aw + m * av 
        Vec3f.scaleAdd(vdest,3,I[m],a,3);
    }
    
    
    /**
     * vdest = I*a 
     */
    public static void transformSpatialVec(float vdest[], int dstIndex, float I[], int iIndex, float a[], int aIndex)
    {
        //fv(I*aw + h x av, m*av - h x aw) 
        
        //I * aw
        Mat3f.transform(I, iIndex, vdest, dstIndex, a, aIndex);
        
        //h x av
        Vec3f.cross(vdest, dstIndex+3, I,iIndex+h, a,aIndex+3);
        
        //I * aw + h x av
        Vec3f.add(vdest,dstIndex,vdest,dstIndex+3);
        
        //h x aw
        Vec3f.cross(vdest,dstIndex+3,I,iIndex+h,a,aIndex);
        
        //-h x aw
        Vec3f.scale(-1f, vdest,dstIndex+3);
        
        //-hx x aw + m * av
        Vec3f.scaleAdd(vdest,dstIndex+3,I[iIndex+m],a,aIndex+3);
    }
}
