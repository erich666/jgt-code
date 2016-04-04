#ifndef DQUEUE_H
#define DQUEUE_H


template <class T> class DQUEUE
		{
		public:
			      inline 		DQUEUE();
			      inline 	       ~DQUEUE();	
			      inline T    	Get();
			      inline T          Head();
			      inline void 	Put(T);
			      T   		Remove(int);
			      inline int  	Count(void);
			      int  		Contains(T);		//idx if contains, else -1
			      void 		Flush();

		private:			//Fwd-declare private friend as private
			struct CELL;
		public:
			
			class Iterator {
				       public:
				       
				       	   inline       Iterator(DQUEUE&);
				       	   inline void  init(DQUEUE&);	 
				       	   inline T&    get();
				       	   inline void  next();
				       	   inline int   left();   
				       	   
				       private:
				       
				       	   CELL* crt;	   
				       };	   	

		private:
		       
			      typedef struct CELL
				       {
					 T            x;
					 struct CELL* next;
					 CELL(T y): x(y),next(0) {}	 
				       } CELL;
				
			      CELL* head;
			      CELL* tail;
			      int   count;
		 };


template <class T> inline DQUEUE<T>::DQUEUE():head(0),tail(0),count(0)
{  }

template <class T> inline DQUEUE<T>::~DQUEUE()
{ Flush(); }

template <class T> void DQUEUE<T>::Flush()
{
  while (head)
  {
    CELL* tmp = head;
    head      = head->next;
    delete tmp;
  }
  count = 0; tail = 0;
}

template <class T> inline int DQUEUE<T>::Count(void)
{  return count;  }


template <class T> inline void DQUEUE<T>::Put(T x)
{
  CELL* tmp = new CELL(x);

  if (!count++)
    tail = head = tmp;
  else
  {
    tail->next = tmp;
    tail       = tmp;
  }
}



template <class T> inline T DQUEUE<T>::Get()
{
  CELL* tmp = head;
  head      = head->next;

  if (!--count) tail = 0;

  T ret = tmp->x;
  delete tmp;
  return ret;
}

template <class T> inline T DQUEUE<T>::Head()
{  return head->x;  }

template <class T> int DQUEUE<T>::Contains(T t)
{
  int i=0;
  for(CELL* p=head;p;p=p->next)
    if (p->x == t) return i; else i++;
  return -1;
}

template <class T> inline DQUEUE<T>::Iterator::Iterator(DQUEUE<T>& q): crt(q.head)
{  }

template <class T> inline void DQUEUE<T>::Iterator::next()
{  crt=crt->next;  }

template <class T> inline int DQUEUE<T>::Iterator::left()
{  return crt != 0;  }

template <class T> inline T& DQUEUE<T>::Iterator::get()
{  return crt->x;  }

template <class T> T DQUEUE<T>::Remove(int i)
{  
   if (!i) return Get();
   else
   {
      CELL *last,*p;
      for(p=head;i;i--) { last = p; p = p->next; }
      last->next = p->next;
      if (p==tail) tail = last;
      T ret = p->x;
      count--; 
      delete p;
      return ret;
   }
}

template <class T> inline void DQUEUE<T>::Iterator::init(DQUEUE<T>& q)
{ crt = q.head; }

#endif


