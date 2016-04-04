#ifndef QUEUE_H
#define QUEUE_H

#include <stdlib.h>

template <class T> class QUEUE
		  {
			 public:
				inline QUEUE(int maxsize);
				inline ~QUEUE();
				inline void Put(T ptr);
				inline T    Get();
				inline int  Count();
				inline int  Contains(T ptr);	   // Returns pos of ptr in queue or -1 if not there	 
				inline T    operator[] (int n);    // Returns ith element in queue starting from beginning
				inline void Flush();
			 private:
				int g;
				int p;
				int count;
				int maxsize;
				T* queue;
		  };

template <class T> inline QUEUE<T>::QUEUE(int Maxsize)
				   :g(0),p(0),count(0),maxsize(Maxsize),queue(new T[Maxsize])
{  }

template<class T> inline QUEUE<T>::~QUEUE()
{  delete[] queue;  }

template<class T> inline T QUEUE<T>::Get()
{
  T ret = queue[g++];
  if (g==maxsize) g = 0;
  count--;
  return ret;
}

template<class T> inline void QUEUE<T>::Put(T ptr)
{
  queue[p++] = ptr;
  if (p==maxsize) p = 0;
  count++;
}

template<class T> inline int QUEUE<T>::Count()
{  return count;  }

template<class T> inline int QUEUE<T>::Contains(T ptr)
{
  int i = g;
  while(i!=p)
  {
	 if (queue[i++]==ptr) return i;
	 if (i==maxsize) i=0;
  }
  return -1;
}




template<class T> inline T QUEUE<T>::operator[](int i)
{  return queue[(g+i)%maxsize];  }


template<class T> inline void QUEUE<T>::Flush()
{  p = g = count = 0;  }

#endif
