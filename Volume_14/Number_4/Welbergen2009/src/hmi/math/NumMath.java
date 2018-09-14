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
 * Numerical functions on data lists
 * @author welberge
 */
public class NumMath
{
    /**
     * Differentiates the values x in the array, assumes equidistant data in x, with distance h
     * assumes dst.length >= x.length-1
     * dst[0]=(x[1]-x[0])/h
     * dst[i]=(x[i+1]-x[i-1])/2*h      
     */
    public static void diff(double[] dst, double[]x, double h)
    {
        for(int i=1;i<x.length-1;i++)
        {
            dst[i]= (x[i+1]-x[i-1])/(2*h);
        }
        dst[0]=(x[1]-x[0])/h;        
    }
    
    /**
     * Differentiates the values x in the array twice, assumes equidistant data in x with distance h
     * assumes dst.length >= x.length-1
     * dst[i]= (x[i+1]-2*x[i]+x[i-1])/(2*h*h), for 1 <= i <= x.length-1
     * dst[0] does not contain a valid value for the 2nd derivative
     */
    public static void diff2(double[] dst, double[]x, double h)
    {
        for(int i=1;i<x.length-1;i++)
        {
            dst[i]= (x[i+1]-2*x[i]+x[i-1])/(h*h);
        }
    }
    
    
    /**
     * Numerical differentiation of x(t)
     * diff = (x(t+h)-x(t-h))/h
     * @param xPrev x(t-h)
     * @param xNext x(t+h)
     * @param h distance 
     * @return dx/dt
     */
    public static float diff(float xPrev, float xNext, float h)
    {
        return (xNext-xPrev)/(2*h);
    }
    
    /**
     * Numerical differentiation of x(t)
     * diff = (x(t+h)-x(t-h))/h
     * @param xPrev x(t-h)
     * @param xNext x(t+h)
     * @param h distance 
     * @return dx/dt
     */
    public static double diff(double xPrev, double xNext, double h)
    {
        return (xNext-xPrev)/(2*h);
    }
    
    /**
     * differerentiates x twice
     * diff2 = (x(t+h)-2*x(t)+x(t-h))/(h*h)
     * @param xPrev x(t-h)
     * @param xCurr x(t)
     * @param xNext x(t+h)
     * @param h distance 
     * @return d2x/dt2
     */    
    public static float diff2(float xPrev, float xCurr, float xNext, float h)
    {
        return (xNext-2*xCurr+xPrev)/(h*h);
    }
    
    /**
     * differerentiates x twice
     * diff2 = (x(t+h)-2*x(t)+x(t-h))/(h*h)
     * @param xPrev x(t-h)
     * @param xCurr x(t)
     * @param xNext x(t+h)
     * @param h distance 
     * @return d2x/dt2
     */    
    public static double diff2(double xPrev, double xCurr, double xNext, double h)
    {
        return (xNext-2*xCurr+xPrev)/(h*h);
    }
    
    /**
     * Differentiates the values x in the array, assumes equidistant data in x, with distance h
     * assumes dst.length >= x.length-1
     * dst[0]=(x[1]-x[0])/h
     * dst[i]=(x[i+1]-x[i-1])/2*h      
     * Assumes the data is aligned in blocks of width, differentiates the block elements seperately 
     */
    public static void diff(double[] dst, double[]x, double h, int width)
    {
        for(int i=1;i<x.length/width-1;i++)
        {
            for(int j=0;j<width;j++)
            {
                dst[i*width+j]= (x[(i+1)*width+j]-x[(i-1)*width+j])/(2*h);
            }
        }
        for(int j=0;j<width;j++)
        {
            dst[j]=(x[width+j]-x[j])/h;
        }
    }
    
    /**
     * Differentiates the values x in the array, assumes equidistant data in x, with distance h
     * assumes dst.length >= x.length-1
     * dst[0]=(x[1]-x[0])/h
     * dst[i]=(x[i+1]-x[i-1])/2*h      
     * Assumes the data is aligned in blocks of width, differentiates the block elements seperately 
     */
    public static void diff(float[] dst, float[]x, float h, int width)
    {
        //(x[i+1]-x[i-1])/2*h
        for(int i=1;i<x.length/width-1;i++)
        {
            for(int j=0;j<width;j++)
            {
                dst[i*width+j]= (x[(i+1)*width+j]-x[(i-1)*width+j])/(2*h);
            }
        }
        
        //dst[0]=(x[1]-x[0])/h
        for(int j=0;j<width;j++)
        {
            dst[j]=(x[width+j]-x[j])/h;
        }
    }
    
    /**
     * Differentiates the values x in the array twice, assumes equidistant data in x with distance h
     * assumes dst.length >= x.length-1
     * dst[i]= (x[i+1]-2*x[i]+x[i-1])/(h*h), for 1 <= i <= x.length-1
     * dst[0]=dst[1]
     * Assumes the data is aligned in blocks of width, differentiates the block elements seperately
     */
    public static void diff2(double[] dst, double[]x, double h, int width)
    {
        //dst[i]= (x[i+1]-2*x[i]+x[i-1])/(h*h)
        for(int i=1;i<x.length/width-1;i++)
        {
            for(int j=0;j<width;j++)
            {
                dst[i*width+j]= (x[(i+1)*width+j]-2*x[i*width+j]+x[(i-1)*width+j])/(h*h);
            }
        }
        
        //dst[0]=dst[1]
        for(int i=0;i<width;i++)
        {
            dst[i]=dst[width+i];
        }
    }
    
    /**
     * Differentiates the values x in the array twice, assumes equidistant data in x with distance h
     * assumes dst.length >= x.length-1
     * dst[i]= (x[i+1]-2*x[i]+x[i-1])/(h*h), for 1 <= i <= x.length-1
     * dst[0]=dst[1]
     * Assumes the data is aligned in blocks of width, differentiates the block elements seperately
     */
    public static void diff2(float[] dst, float[]x, float h, int width)
    {
        for(int i=1;i<x.length/width-1;i++)
        {
            for(int j=0;j<width;j++)
            {
                dst[i*width+j]= (x[(i+1)*width+j]-2*x[i*width+j]+x[(i-1)*width+j])/(h*h);
            }
        }
        for(int i=0;i<width;i++)
        {
            dst[i]=dst[width+i];
        }
    }
    /**
     * Linear interpolation of equadistantly placed points, the distance between the points is h:
     * x(t) = x[i] + (x(i)-x(i+1)/h * (t-i*h) with i*h<=t, (i+h)*h>=t 
     */
    public static double interpolate(double x[], double h, double t)
    {
        int i = (int)(t/h);
        
        //give boundary values if boundaries are crossed
        if(i>x.length-1)
        {
            return x[x.length-1];
        }
        if(i<0)
        {    
            return x[0];
        }
        
        return (t-i*h) * (x[i+1]-x[i])/h  + x[i];        
    }
    
    /**
     * Linear interpolation of equadistantly placed points, the distance between the points is h:
     * x(t) = x[i] + (x(i)-x(i+1)/h * (t-i*h) with i*h<=t, (i+h)*h>=t 
     */
    public static float interpolate(float x[], float h, float t)
    {
        int i = (int)(t/h);
        
        //give boundary values if boundaries are crossed
        if(i>x.length-1)
        {
            return x[x.length-1];
        }
        if(i<0)
        {    
            return x[0];
        }
        
        return (t-i*h) * (x[i+1]-x[i])/h  + x[i];        
    }
    
    /**
     * Linear interpolation of equadistantly placed points, the distance between the points is h:
     * x(t) = x[i] + (x(i)-x(i+1)/h * (t-i*h) with i*h<=t, (i+h)*h>=t  
     */
    public static void interpolate(double dst[], double x[], double h, double t)
    {
        int i = (int)(t/h);
        int width = dst.length;
        
        //give boundary values if boundaries are crossed
        if(i>x.length/width-width)
        {
            System.arraycopy(x, x.length-width, dst, 0, width);
            return;
        }
        if(i<=0)
        {    
            System.arraycopy(x, 0, dst, 0, width);
            return;
        }        
        
        for(int j=0;j<width;j++)
        {
            int k = i*width+j;
            dst[j] = (t-i*h) * (x[k+width]-x[k])/h + x[k];
        }            
    }
    
    /**
     * Linear interpolation of equadistantly placed points, the distance between the points is h:
     * x(t) = x[i] + (x(i)-x(i+1)/h * (t-i*h) with i*h<=t, (i+h)*h>=t  
     */
    public static void interpolate(float dst[], float x[], float h, float t)
    {
        int i = (int)(t/h);
        int width = dst.length;
        
        //give boundary values if boundaries are crossed
        if(i>x.length/width-width)
        {
            System.arraycopy(x, x.length-width, dst, 0, width);
            return;
        }
        if(i<0)
        {    
            System.arraycopy(x, 0, dst, 0, width);
            return;
        }        
        
        for(int j=0;j<width;j++)
        {
            int k = i*width+j;
            dst[j] = (t-i*h) * (x[k+width]-x[k])/h + x[k];
        }            
    }
}
