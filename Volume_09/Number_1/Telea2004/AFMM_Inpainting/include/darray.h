#ifndef DARRAY_H
#define DARRAY_H

/* DARRAY.H     A dynamic array class. The array expands if necessary */
/*		and elements can be randomly deleted from it	      */	 



template <class T, int GROW_SIZE = 20> class DARRAY	  //DARRAY is parametrized by a GROW_SIZE factor
			 {				  //which has a template-default value.	
			 public:

			     DARRAY(int N=GROW_SIZE);	  //USERDEF constr: N-positions array
			     DARRAY(const DARRAY&);	  //COPY constr
			     inline void Init(int N);	  //Reallocs to N positions (deletes old array)
			     inline void InitC(int);      //Calls Init + SetCount
			     inline ~DARRAY();	
			     inline T& operator[](int) const;   //Subscript: no range checking
			     inline T& operator[](int);   //Subscript: no range checking
			     DARRAY& 
  			     operator=(const DARRAY&);    //Assignment of arrays
  			     DARRAY&
  			     operator=(const T&);	  //Fills array up to Count() with given elem		
			     void Delete(int);    	  //Delete i-th array element, shrink array 
			     void Insert(int,T);  	  //Insert element at i-th position, expand if case
							  // (if int==-1, add elem at array's end, like Add())
			     inline int Count() const;	  //Return number of elements added to array 
			     int  Remove(T);	  	  //Search and remove an element. Returns
						  	  //  position that element had in array or -1
			     int  RemoveAll(T);		  //Search and remove all occurences of an element.
							  //  Returns # removed elements.
			     inline void Add(T);	  //Adds an elem at end (expands if case)
			     inline void operator+=(T);   //Same as Add, but won't check for expansion need
			     inline void AddOrd(T);	  //Adds an element in order in the array (assumes '<')
							  //	(doesn't add if element already in there)
			     inline int Append(T);	  //Same as Add() but done only if element not already in
							  //  (provided for efficiency vs !Contains()+Add() )
							  // Returns 0 if element already in array, else 1	
			     int Contains(T) const; 	  //Checks if elem is in array (returns pos/-1) 
			     int FindOrd(T) const;	  //As above, but assumes the array is ordered.	
			     inline void Flush();	  //Flushes all elems in the array (size stays the same) 
			     void SetCount(int);	  //Set array counter (NOT recommended!)
			      
			     	
			  protected:

			     T*   a;
			     int  n;
			     int sz;	
  				
			     inline void add_and_grow(T); //utility for growing array & adding an elem to it
	   		  }; 


template <class T,int GROW_SIZE> inline DARRAY<T,GROW_SIZE>::DARRAY(int N)
	:a(new T[N]),n(0),sz(N)
{  }

template <class T,int GROW_SIZE> DARRAY<T,GROW_SIZE>::DARRAY(const DARRAY& arr)
	:a(new T[arr.sz]),n(arr.n),sz(arr.sz)
{  for (T *i=a,*end=a+sz,*av=arr.a; i < end;i++,av++) *i = *av;  }

template <class T,int GROW_SIZE> inline DARRAY<T,GROW_SIZE>::~DARRAY()
{  delete[] a;  }

template <class T,int GROW_SIZE> 
DARRAY<T,GROW_SIZE>& DARRAY<T,GROW_SIZE>::operator=(const DARRAY<T,GROW_SIZE>& arr)
{
  if (&arr == this) return *this;
  if (sz!=arr.sz)
  {  delete[] a;  a   = new T[sz=arr.sz];  }
  n = arr.n;
  
  for (T *i=a,*end=a+sz,*av=arr.a; i < end; i++,av++) *i = *av;
  
  return *this;
}


template <class T,int GROW_SIZE> 
DARRAY<T,GROW_SIZE>& DARRAY<T,GROW_SIZE>::operator=(const T& t)
{
  for (T *i=a,*end=a+n; i < end; i++) *i = t;  
  return *this;
}


template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::Init(int N)
{  delete[] a;  a = new T[sz=N]; n = 0;  }


template <class T,int GROW_SIZE> inline int DARRAY<T,GROW_SIZE>::Count() const
{  return n;  }

template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::SetCount(int c)
{  n = c;  }

template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::InitC(int N)
{  Init(N); SetCount(N);  }

template <class T,int GROW_SIZE> inline T& DARRAY<T,GROW_SIZE>::operator[](int i) const
{  return a[i];  }

template <class T,int GROW_SIZE> inline T& DARRAY<T,GROW_SIZE>::operator[](int i) 
{  return a[i];  }

#define INSERT(i,x)	/*defined as such to ensure inlining*/			\
{										\
  if (n<sz)		/*array has space enough for new element*/		\
  {										\
    for(register int j=n-1;j>=i;j--) a[j+1] = a[j];				\
    a[i] = x; 									\
  }										\
  else			/*array full, must expand*/				\
  {										\
    T* tmp = new T[sz+GROW_SIZE];						\
    register int j; for(j=0;j<i;j++) tmp[j] = a[j];				\
    tmp[i] = x;									\
    for(j=sz;j>i;j--) tmp[j] = a[j-1];						\
    delete[] a;									\
    sz += GROW_SIZE;								\
    a   = tmp;									\
  }										\
  n++;										\
}


template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::Insert(int i,T x)
{
  if (i==-1) Add(x);
  else       INSERT(i,x);
}


template <class T,int GROW_SIZE> void DARRAY<T,GROW_SIZE>::Delete(int i)
{
  if (n)
  {
    n--;  
    for(register int j=i;j<n;j++)   a[j] = a[j+1];
  }
}


template <class T,int GROW_SIZE> inline int DARRAY<T,GROW_SIZE>::Remove(T x)
{
  int i,ret; 
  if (n==0) return -1;			//empty array, remove fails

  for(i=0;i<n;i++) if (a[i]==x) break;  //search for x
  if (i==n) return -1;			//x not found, remove fails
  ret = i; 				//x found at position i
  n--;					//one less element
  for(;i<n;i++)  a[i] = a[i+1];		//SHL array from i to end	
  return ret;				//return pos of removed element
}


template <class T,int GROW_SIZE> inline int DARRAY<T,GROW_SIZE>::RemoveAll(T x)
{  int c; for(c=0;Remove(x)!=-1;) c++; return c;  }


template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::Add(T x)
{
  if (n<sz)				//space enough to add new element
    a[n] = x;				
  else					//array full, grow and add element x
    add_and_grow(x);
  n++;
}


template <class T,int GROW_SIZE> inline void  DARRAY<T,GROW_SIZE>::operator+=(T x)
{
  a[n++] = x;				//REMARK: this method assumes there's enough space in array
}

template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::AddOrd(T x)
{
   if (!n) { Add(x); return; }		//empty array: just intsert   
   
   register int l = 0, r = n-1;
   do {
	register int m = (l+r)>>1;
	register T   y = a[m];	
      	if (x==y)   return; 		//found x at position 'm', just return
        if (x<y)    r = --m;
	else	    l = ++m;
      }
   while (l<=r);
   
   INSERT(l,x);  			//x not found: insert x at position l
}

template <class T,int GROW_SIZE> inline int DARRAY<T,GROW_SIZE>::FindOrd(T x) const
{
   if (n)				//empty array: no search
   {
     register int l = 0, r = n-1;
     do {
 	register int m = (l+r)>>1;
	register T   y = a[m];	
      	if (x==y)   return m; 		//found x at position 'm'
        if (x<y)    r = --m;
	else	    l = ++m;
      }
     while(l<=r);
   }

   return -1;				//found nothing
}

template <class T,int GROW_SIZE> inline int DARRAY<T,GROW_SIZE>::Append(T x)
{
  for(int i=0;i<n;i++) if (a[i]==x) return 0;  //element already in there
  Add(x);	
  return 1;				       //element was added	
}


template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::add_and_grow(T x)
//REMARK: this routine is called by above Add(). Putting it separately allows inlining Add(),
//	  which is important since add_and_grow() won't be called in most Add() cases. 
{
    T* tmp = new T[sz+GROW_SIZE];
    for(int j=0;j<sz;j++) tmp[j] = a[j];
    tmp[sz] = x;
    delete[] a;
    a    =  tmp;
    sz  += GROW_SIZE;
}






template <class T,int GROW_SIZE> int DARRAY<T,GROW_SIZE>::Contains(T x) const
{
  for(int i=0;i<n;i++) if (a[i]==x) return i;
  return -1;
}

template <class T,int GROW_SIZE> inline void DARRAY<T,GROW_SIZE>::Flush()
{  n = 0;  }


#endif
