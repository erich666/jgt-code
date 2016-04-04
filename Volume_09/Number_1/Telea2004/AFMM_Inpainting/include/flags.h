#ifndef FLAGS_H
#define FLAGS_H

// FLAGS:	Implements the flagging of the field's cells for the narrowband representation.
//		Every field cell used in the fast marching method can be:	
//
//		ALIVE:		its field value is known
//		NARROW_BAND:	its value is in the narrow band, i.e. it is still in the updating process
//		FAR_AWAY:	its value is not yet known, i.e. has not yet entered the narrow band
//		EXTREMUM:	this is a special case of known values. Extremum values are known values which
//				are also detected as extremum points of the constructed signal.
//		
//		The ctor of FLAGS constructs this from one or several fields and also adjusts these fields
//		to be evolved by the fast marching method, as follows:
//
//		FLAGS(f,low):	 Evolution starts from and inside curve where f == low, if low > 0.
//				 Evolution starts from and outside curve where f == -low, if low < 0.
//				 The evolved signal is set to 0 outside the evolved region, 1 on the initial curve,
//				 and grows from 1 monotonically in the evolved region.
//
//		FLAGS(f,g,low):  As above, but the evolved signal is set to g on the initial curve and outside 
//				 the evolved region, and INFINITY inside the evolved region. The signal grows then
//				 not from 1, as above, but from g's values on the initial curve.
//

#include "field.h"


class FLAGS : public FIELD<int>
	{
	public:
	
		enum FLAG_TYPE { NARROW_BAND,ALIVE,FAR_AWAY,EXTREMUM };


		     FLAGS(FIELD<float>& f,float low);		//Ctor. See info above. 
		     FLAGS(FIELD<float>& f,			//Ctor. See info above.
			   const FIELD<float>& t,float low);

		int  alive(int i,int j) const 		{ return value(i,j)==ALIVE; }
		int  narrowband(int i,int j) const	{ return value(i,j)==NARROW_BAND; }
		int  faraway(int i,int j) const		{ return value(i,j)==FAR_AWAY;  }
		int  extremum(int i,int j) const     	{ return value(i,j)==EXTREMUM; }
		int  connected(int i,int j) const;
	FIELD<int>*  tagConnect(int* =0) const;	//produce connectivity-field from this, return # produced clusters if desired
	        void writeRGBCodedPPM(char*);	//write this as a special color-coded PPM file

	private:
	
	int 	flood_fill(FIELD<int>&,int i,int j,int val) const;
	
	};

#endif

