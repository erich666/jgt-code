/*     The bi-directional generating algorithm for rational quadratic Bezier curves  */
/*                                             and rational cubic Bezier curves      */
/*     The program can be run in Microsoft VC6.0 development environment             */
/*     Written by Zhong Li, 18/2/2005                                                */
/*     Here are two functions: DrawQuadratic() and DrawCubic()                       */
//     DrawQuadratic():
void DrawQuadratic(CPoint p0,CPoint p1,CPoint p2,double w0,double w1,double w2)
{
	double wbar;
	CPoint pn,pp;
	double max1,max2,max3x,max3y,fd,gd;  
	int n;
	wbar=w1/sqrt(w0*w2);                
	max1=(sqrt(w2/w0)>sqrt(w0/w2))?sqrt(w2/w0):sqrt(w0/w2);
	max2=(wbar>(2/(1+wbar)))?wbar:2/(1+wbar);
	max3x=fabs(p2.x-p1.x)>fabs(p1.x-p0.x)?fabs(p2.x-p1.x):fabs(p1.x-p0.x);
	max3y=fabs(p2.y-p1.y)>fabs(p1.y-p0.y)?fabs(p2.y-p1.y):fabs(p1.y-p0.y);
	fd=2*max3x*max2*max1;
	gd=2*max3y*max2*max1;
	n=(int)((fd>gd)?fd:gd);              //Get n valve by Hermann's estimate
	double D1ui,D2ui,D1ri,D2ri,D1vi,D2vi,B1uj,B1rj,B1vj;
	D2ui=2*w0*p0.x-4*w1*p1.x+2*w2*p2.x;  // get 2 order forward difference constant
	D2ri=2*w0*p0.y-4*w1*p1.y+2*w2*p2.y;
	D2vi=2*w0-4*w1+2*w2;                               
	double ui,vi,ri,ui1,vi1,ri1,xi,xi1,zi,zi1,yi,yi1,qi,qi1,xia,xib,xic,yia,yib,yic;
	double uj,vj,rj,uj1,vj1,rj1,xj,xj1,zj,zj1,yj,yj1,qj,qj1,xja,xjb,xjc,yja,yjb,yjc;
	double Judge;
	int i,j;
	CTime tstart=CTime::GetCurrentTime();  // Start time test
	for(int k=0;k<10000;k++)               // for 10000 times
	{
		xi=p0.x;zi=0;
		yi=p0.y;qi=0;
		pn.x=(int)xi;pn.y=(int)yi;
		pDC->SetPixel(pn,RGB(0,0,255));   //draw the first pixel
		xia=xi;yia=yi;
		xib=xi;yib=yi;                    //using by geting rid of corner point
		xj=p2.x;zj=0;
		yj=p2.y;qj=0;
		pp.x=(int)xj;pp.y=(int)yj;
		pDC->SetPixel(pp,RGB(0,0,255));   //draw the end pixel
		xja=xj;yja=yj;
		xjb=xj;yjb=yj;                    //using by geting rid of corner point
		ui=n*n*w0*p0.x;             // compute initial xi,zi,yi,qi in one direction
		ri=n*n*w0*p0.y;
		vi=n*n*w0;
		uj=n*n*w2*p2.x;             //compute initial xj,zj,yj,qj in another direction
		rj=n*n*w2*p2.y;
		vj=n*n*w2;
		i=1;
		j=n-1;
		while(fabs(pn.x-pp.x)>1||fabs(pn.y-pp.y)>1)
		{                               //get ui+1 by forward difference
			D1ui=(1+2*i-2*n)*w0*p0.x+2*(n-2*i-1)*w1*p1.x+(1+2*i)*w2*p2.x; 
			ui1=ui+D1ui-D2ui;
			D1ri=(1+2*i-2*n)*w0*p0.y+2*(n-2*i-1)*w1*p1.y+(1+2*i)*w2*p2.y;
			ri1=ri+D1ri-D2ri;           //get ri+1 by forward difference
			D1vi=(1+2*i-2*n)*w0+2*(n-2*i-1)*w1+(1+2*i)*w2;
			vi1=vi+D1vi-D2vi;           //get vi+1 by forward difference
			Judge=ui1/vi1-ui/vi-zi/vi;
			if(Judge<-0.5)                     //using recursive relation to get xi+1,zi+1
			{
				xi1=xi-1;
				zi1=zi+xi*(vi1-vi)+ui-ui1-vi1;
			}
			else if(Judge>0.5)
			{
				xi1=xi+1;
				zi1=zi+xi*(vi1-vi)+ui-ui1+vi1;
			}
			else
			{
				xi1=xi;
				zi1=zi+xi*(vi1-vi)+ui-ui1;
			}
			Judge=ri1/vi1-ri/vi-qi/vi;
			if(Judge<-0.5)                    //using recursive relation to get yi+1,qi+1
			{
				yi1=yi-1;
				qi1=qi+yi*(vi1-vi)+ri-ri1-vi1;
			}
			else if(Judge>0.5)
			{
				yi1=yi+1;
				qi1=qi+yi*(vi1-vi)+ri-ri1+vi1;
			}
			else
			{	
				yi1=yi;
				qi1=qi+yi*(vi1-vi)+ri-ri1;
			}
			xic=xi1;yic=yi1;
			if(fabs(xic-xia)>1||fabs(yic-yia)>1)
			{                             //using simple judgement to get rid of corner point
				pn.x=(int)xib;pn.y=(int)yib;
				pDC->SetPixel(pn,RGB(0,0,255));
				xia=xib;yia=yib;
				xib=xic;yib=yic;
			}
			else
			{
				xib=xic;yib=yic;
			}
			xi=xi1;yi=yi1;
			ui=ui1;ri=ri1;vi=vi1;
			zi=zi1;qi=qi1;
			i++;
                                                 //Draw another direction to render
			B1uj=(2*j-2*n-1)*w0*p0.x+2*(1-2*j+n)*w1*p1.x+(2*j-1)*w2*p2.x;
			uj1=uj-B1uj-D2ui;            //get ui-1 by backward difference 
			B1rj=(2*j-2*n-1)*w0*p0.y+2*(1-2*j+n)*w1*p1.y+(2*j-1)*w2*p2.y;
			rj1=rj-B1rj-D2ri;            //get ri-1 by backward difference
			B1vj=(2*j-2*n-1)*w0+2*(1-2*j+n)*w1+(2*j-1)*w2;
			vj1=vj-B1vj-D2vi;            //get vi-1 by backward difference
			Judge=uj1/vj1-uj/vj-zj/vj;
			if(Judge<-0.5)               //using recursive relation to get xj-1,zj-1
			{
				xj1=xj-1;
				zj1=zj+xj*(vj1-vj)+uj-uj1-vj1;
			}
			else if(Judge>=0.5)
			{
				xj1=xj+1;
				zj1=zj+xj*(vj1-vj)+uj-uj1+vj1;
			}
			else
			{
				xj1=xj;
				zj1=zj+xj*(vj1-vj)+uj-uj1;
			}
			Judge=rj1/vj1-rj/vj-qj/vj;
			if(Judge<-0.5)                     //using recursive relation to get yj-1,qj-1
			{
				yj1=yj-1;
				qj1=qj+yj*(vj1-vj)+rj-rj1-vj1;
			}
			else if(Judge>=0.5)
			{
				yj1=yj+1;
				qj1=qj+yj*(vj1-vj)+rj-rj1+vj1;
			}
			else
			{
				yj1=yj;
				qj1=qj+yj*(vj1-vj)+rj-rj1;
			}
			xjc=xj1;yjc=yj1;
			if(fabs(xjc-xja)>1||fabs(yjc-yja)>1) 
			{                           //using simple judgement to get rid of corner point
				pp.x=(int)xjb;pp.y=(int)yjb;
				pDC->SetPixel(pp,RGB(0,0,255));
				xja=xjb;yja=yjb;
				xjb=xjc;yjb=yjc;
			}
			else
			{
				xjb=xjc;yjb=yjc;
			}
			xj=xj1;yj=yj1;	
			uj=uj1;rj=rj1;vj=vj1;
			zj=zj1;qj=qj1;
			j--;
		}
	}
	CTime tend=CTime::GetCurrentTime();    // get finishing time 
	CTimeSpan	elapse=tend-tstart;
	CString str;
	str.Format("elapsed time is: %d second!",elapse.GetSeconds());
	pDC->TextOut(300,30,str);
}
//    DrawCubic():
void DrawCubic(CPoint p0,CPoint p1,CPoint p2,CPoint p3,double w0,double w1,double w2,double w3)
{
	double Q,q,max1,min1,max2,min2,maxx,maxy,fd,gd;  
	int n;
	max1=w0>w1?w0:w1;
	max2=w2>w3?w2:w3;
	Q=max1>max2?max1:max2;
	min1=w0>w1?w1:w0;
	min2=w2>w3?w3:w2;
	q=min1>min2?min2:min1;
	maxx=fabs(p2.x-p1.x)>fabs(p1.x-p0.x)?fabs(p2.x-p1.x):fabs(p1.x-p0.x);
	maxx=maxx>fabs(p3.x-p2.x)?maxx:fabs(p3.x-p2.x);
	maxy=fabs(p2.y-p1.y)>fabs(p1.y-p0.y)?fabs(p2.y-p1.y):fabs(p1.y-p0.y);
	maxy=maxy>fabs(p3.y-p2.y)?maxy:fabs(p3.y-p2.y);
	fd=3*Q*Q*maxx/(q*q);
	gd=3*Q*Q*maxy/(q*q);
	n=(int)((fd>gd)?fd:gd);              //Get n valve by Floater's estimate	
	double ui,vi,ri,ui1,vi1,ri1,xi,xi1,zi,zi1,yi,yi1,qi,qi1,xia,xib,xic,yia,yib,yic;
	double uj,vj,rj,uj1,vj1,rj1,xj,xj1,zj,zj1,yj,yj1,qj,qj1,xja,xjb,xjc,yja,yjb,yjc;
	double Judge,pos;
	int i,j;
	CTime tstart=CTime::GetCurrentTime();  // Start time test
	for(int k=0;k<10000;k++)               // for 10000 times
	{
		xi=p0.x;zi=0;
		yi=p0.y;qi=0;
		pn.x=(int)xi;pn.y=(int)yi;
		pDC->SetPixel(pn,RGB(0,0,255));   //draw the first pixel
		xia=xi;yia=yi;
		xib=xi;yib=yi;                    //using by geting rid of corner point
		xj=p3.x;zj=0;
		yj=p3.y;qj=0;
		pp.x=(int)xj;pp.y=(int)yj;
		pDC->SetPixel(pp,RGB(0,0,255));   //draw the end pixel
		xja=xj;yja=yj;
		xjb=xj;yjb=yj;                    //using by geting rid of corner point
		ui=w0*p0.x;             // compute initial xi,zi,yi,qi in one direction
		ri=w0*p0.y;
		vi=w0;
		uj=w3*p3.x;             //compute initial xj,zj,yj,qj in another direction
		rj=w3*p3.y;
		vj=w3;
		i=1;
		j=n-1;
		while(fabs(pn.x-pp.x)>1||fabs(pn.y-pp.y)>1)
		{
			pos=double(i)/n;
			ui1=(1-pos)*(1-pos)*(1-pos)*w0*p0.x+3*pos*(1-pos)*(1-pos)*w1*p1.x
				+3*pos*pos*(1-pos)*w2*p2.x+pos*pos*pos*w3*p3.x;
			ri1=(1-pos)*(1-pos)*(1-pos)*w0*p0.y+3*pos*(1-pos)*(1-pos)*w1*p1.y
				+3*pos*pos*(1-pos)*w2*p2.y+pos*pos*pos*w3*p3.y;
			vi1=(1-pos)*(1-pos)*(1-pos)*w0+3*pos*(1-pos)*(1-pos)*w1
				+3*pos*pos*(1-pos)*w2+pos*pos*pos*w3;
			Judge=ui1/vi1-ui/vi-zi/vi;
			if(Judge<-0.5)                     //using recursive relation to get xi+1,zi+1
			{
				xi1=xi-1;
				zi1=zi+xi*(vi1-vi)+ui-ui1-vi1;
			}
			else if(Judge>0.5)
			{
				xi1=xi+1;
				zi1=zi+xi*(vi1-vi)+ui-ui1+vi1;
			}
			else
			{
				xi1=xi;
				zi1=zi+xi*(vi1-vi)+ui-ui1;
			}
			Judge=ri1/vi1-ri/vi-qi/vi;
			if(Judge<-0.5)                    //using recursive relation to get yi+1,qi+1
			{
				yi1=yi-1;
				qi1=qi+yi*(vi1-vi)+ri-ri1-vi1;
			}
			else if(Judge>0.5)
			{
				yi1=yi+1;
				qi1=qi+yi*(vi1-vi)+ri-ri1+vi1;
			}
			else
			{	
				yi1=yi;
				qi1=qi+yi*(vi1-vi)+ri-ri1;
			}
			xic=xi1;yic=yi1;
			if(fabs(xic-xia)>1||fabs(yic-yia)>1)
			{                             
				pn.x=(int)xib;pn.y=(int)yib;
				pDC->SetPixel(pn,RGB(0,0,255));
				xia=xib;yia=yib;
				xib=xic;yib=yic;
			}
			else
			{
				xib=xic;yib=yic;
			}
			xi=xi1;yi=yi1;
			ui=ui1;ri=ri1;vi=vi1;
			zi=zi1;qi=qi1;
			i++;
                       	                       //Draw another direction to render
			pos=double(j)/n;
			uj1=(1-pos)*(1-pos)*(1-pos)*w0*p0.x+3*pos*(1-pos)*(1-pos)*w1*p1.x
				+3*pos*pos*(1-pos)*w2*p2.x+pos*pos*pos*w3*p3.x;
			rj1=(1-pos)*(1-pos)*(1-pos)*w0*p0.y+3*pos*(1-pos)*(1-pos)*w1*p1.y
				+3*pos*pos*(1-pos)*w2*p2.y+pos*pos*pos*w3*p3.y;
			vj1=(1-pos)*(1-pos)*(1-pos)*w0+3*pos*(1-pos)*(1-pos)*w1
				+3*pos*pos*(1-pos)*w2+pos*pos*pos*w3;
			Judge=uj1/vj1-uj/vj-zj/vj;
			if(Judge<-0.5)                      //using recursive relation to get xj-1,zj-1
			{
				xj1=xj-1;
				zj1=zj+xj*(vj1-vj)+uj-uj1-vj1;
			}
			else if(Judge>=0.5)
			{
				xj1=xj+1;
				zj1=zj+xj*(vj1-vj)+uj-uj1+vj1;
			}
			else
			{
				xj1=xj;
				zj1=zj+xj*(vj1-vj)+uj-uj1;
			}
			Judge=rj1/vj1-rj/vj-qj/vj;
			if(Judge<-0.5)                     //using recursive relation to get yj-1,qj-1
			{
				yj1=yj-1;
				qj1=qj+yj*(vj1-vj)+rj-rj1-vj1;
			}
			else if(Judge>=0.5)
			{
				yj1=yj+1;
				qj1=qj+yj*(vj1-vj)+rj-rj1+vj1;
			}
			else
			{
				yj1=yj;
				qj1=qj+yj*(vj1-vj)+rj-rj1;
			}
			xjc=xj1;yjc=yj1;
			if(fabs(xjc-xja)>1||fabs(yjc-yja)>1) //using simple judgement to get rid of corner point
			{
				pp.x=(int)xjb;pp.y=(int)yjb;
				pDC->SetPixel(pp,RGB(0,0,255));
				xja=xjb;yja=yjb;
				xjb=xjc;yjb=yjc;
			}
			else
			{
				xjb=xjc;yjb=yjc;
			}
			xj=xj1;yj=yj1;	
			uj=uj1;rj=rj1;vj=vj1;
			zj=zj1;qj=qj1;
			j--;
		}
	}
	CTime tend=CTime::GetCurrentTime();    // get finishing time 
	CTimeSpan	elapse=tend-tstart;
	CString str;
	str.Format("elapsed time is: %d second!",elapse.GetSeconds());
	pDC->TextOut(300,30,str);
}