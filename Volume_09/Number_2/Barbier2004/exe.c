#include <stdlib.h>

#include "shapes.h"

/*!
  \mainpage
  This project groups C++ code to compute the distance between 
  a point in space and cylinder, cones, cylinder-spheres and cone-spheres.
*/

double R()
{
  return (rand()%32000)/32000.0;
}

Vector V(const double& r)
{
  Vector v(R(),R(),R());
    
  v-=Vector(0.5);
    
  v*=2.0*r;
  
  return v;
}

Vector p[10000];
Vector d[10000];
double t[10000];

void CheckCylinder()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(3.0);
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double r=R();

    Cylinder cylinder(a,b,r);

    for (int j=0;j<10000;j++)
    {
      double a=cylinder.R(p[j]);      
    }
  }
}

void CheckCone()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(3.0);
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double ra=0.25+R();
    double rb=0.75+R();
    if (ra<rb) { double t=rb; rb=ra; ra=t; }

    Cone cone(a,b,ra,rb);

    for (int j=0;j<10000;j++)
    {
      double a=cone.R(p[j]);      
    }
  }
}

void CheckSphereCone()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(3.0);
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double ra=0.25+R();
    double rb=R();
    if (ra<rb) { double t=rb; rb=ra; ra=t; }

    SphereCone spherecone(a,b,ra,rb);

    for (int j=0;j<10000;j++)
    {
      double a=spherecone.R(p[j]);      
    }
  }
}

void CheckCylinderSphere()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(3.0);
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double r=R();

    SphereCylinder spherecylinder(a,b,r);

    for (int j=0;j<10000;j++)
    {
      double a=spherecylinder.R(p[j]);      
    }
  }
}

void CheckCylinderSphereT()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(1.0);
    d[j]=Normalized(V(1.0));
    t[j]=3.0*R();
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double r=R();

    SphereCylinder spherecylinder(a,b,r);
    spherecylinder.Set(p[i],d[i]);      

    for (int j=0;j<10000;j++)
    {
      double a=spherecylinder.R(t[j]);      
    }
  }
}

void CheckSphereConeT()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(3.0);
    d[j]=Normalized(V(1.0));
    t[j]=3.0*R();
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double ra=0.25+R();
    double rb=R();
    if (ra<rb) { double t=rb; rb=ra; ra=t; }

    SphereCone spherecone(a,b,ra,rb);
    spherecone.Set(p[i],d[i]);      

    for (int j=0;j<10000;j++)
    {
      double a=spherecone.R(t[j]);      
    }
  }
}

void CheckCylinderT()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(1.0);
    d[j]=Normalized(V(1.0));
    t[j]=3.0*R();
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double r=R();

    Cylinder cylinder(a,b,r);
    cylinder.Set(p[i],d[i]);      
    for (int j=0;j<10000;j++)
    {
      double a=cylinder.R(t[j]);  
    }
  }
}

void CheckConeT()
{
  for (int j=0;j<10000;j++)
  {
    p[j]=V(1.0);
    d[j]=Normalized(V(1.0));
    t[j]=3.0*R();
  }

  for (int i=0;i<10000;i++)
  {
    Vector a=V(1.0);
    Vector b=V(1.0);
    
    double ra=0.25+R();
    double rb=0.75+R();
    if (ra<rb) { double t=rb; rb=ra; ra=t; }

    Cone cone(a,b,ra,rb);

    cone.Set(p[i],d[i]);      
    for (int j=0;j<10000;j++)
    {
      double a=cone.R(t[j]);      
    }
  }
}

int main()
{
//  CheckCylinderT();
//  CheckConeT();
  CheckCylinderSphereT();
//  CheckSphereConeT();

  return 0;
}
