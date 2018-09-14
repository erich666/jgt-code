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
package hmi.math.digitalfilter;

/**
 * Butterworth digital low-pass filter
 * @author welberge
 */
public class Butterworth
{
    /**
     * Butterworth-filters the data
     * @param Fin input data
     * @param fc cutt-off frequency
     * @param fs sample frequency
     * @param pass pass nr      
     * @param Fout output data
     */
    public static void butterworth(float Fin[],float fc,float fs,int pass, float Fout[])
    {
        double C = Math.pow(Math.pow(2, 1/pass)-1,0.25);
        double sigmaC = Math.tan(Math.PI*fc/fs)/C;
        double K1 = Math.sqrt(2*sigmaC);
        double K2 = sigmaC*sigmaC;
        double a0 = K2/(1+K1+K2);
        double a1 = 2*a0;
        double a2 = a0;
        double K3 = (2*a0)/K2;
        double b1 = -2*a0+K3;
        double b2 = 1-2*a0-K3;
        int length = Fin.length;
        Fout[0]=Fin[0];
        Fout[1]=Fin[1];
        for (int i=3;i<length;i++)
        {
            Fout[i] = (float)(a0*Fin[i]+a1*Fin[i-1]+a2*Fin[i-2]+b1*Fout[i-1]+b2*Fout[i-2]);
        }
    }
    
    /**
     * Butterworth-filters the data, assumes the input is aligned in blocks of width floats 
     * @param Fin input data
     * @param fc cutt-off frequency
     * @param fs sample frequency
     * @param pass pass nr      
     * @param width block sizes
     * @param Fout output data
     */
    public static void butterworth(float Fin[],float fc,float fs,int pass, int width, float Fout[])
    {
        double C = Math.pow(Math.pow(2, 1.0/pass)-1,0.25);
        double sigmaC = Math.tan(Math.PI*fc/fs)/C;
        double K1 = Math.sqrt(2*sigmaC);
        double K2 = sigmaC*sigmaC;
        double a0 = K2/(1+K1+K2);
        double a1 = 2*a0;
        double a2 = a0;
        double K3 = (2*a0)/K2;
        double b1 = -2*a0+K3;
        double b2 = 1-2*a0-K3;
        int length = Fin.length/width;
        for(int i=0;i<width*2;i++)
        {
            Fout[i]=Fin[i];            
        }
        for (int i=2;i<length;i++)
        {
            for(int j=0;j<width;j++)
            {
                Fout[i*width+j] = (float)(a0*Fin[i*width+j]+a1*Fin[(i-1)*width+j]+a2*Fin[(i-2)*width+j]+b1*Fout[(i-1)*width+j]+b2*Fout[(i-2)*width+j]);
            }
        }
    }
}
