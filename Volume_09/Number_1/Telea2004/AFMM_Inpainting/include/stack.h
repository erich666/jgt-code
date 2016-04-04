#ifndef STACK_H
#define STACK_H


// STACK:		Simple stack implementation based on the DARRAY class


#include "darray.h"




template <class T, int GROW_SIZE = 20> class STACK :	  	  		//STACK is parametrized by a GROW_SIZE factor
			 private DARRAY<T,GROW_SIZE>	  	  		//which has a template-default value.
			 {				  	
			 public:

			     			  STACK(int N=GROW_SIZE);	//USERDEF constr: N-positions stack
			     void 		  Push(T);		  	//Push op
			     T 			  Pop();			//Pop  op
			     T			  Top();			//Returns top of stack		
			     DARRAY<T,GROW_SIZE>::operator[];	  		//Inherited   
			     DARRAY<T,GROW_SIZE>::Count;	  		//Inherited
			     DARRAY<T,GROW_SIZE>::Contains; 	  		//Inherited
			     DARRAY<T,GROW_SIZE>::Flush;	  		//Inherited
			      
			     	
	   		  }; 


template <class T,int GROW_SIZE> inline STACK<T,GROW_SIZE>::STACK(int N): DARRAY<T,GROW_SIZE>(N)
{  }

template <class T,int GROW_SIZE> inline void STACK<T,GROW_SIZE>::Push(T t)
{  Add(t);  }

template <class T,int GROW_SIZE> inline T STACK<T,GROW_SIZE>::Top()
{  return a[n-1];  }

template <class T,int GROW_SIZE> inline T STACK<T,GROW_SIZE>::Pop()
{
  T tmp = a[n-1];
  Delete(n-1);
  return tmp;
}

#endif


  
