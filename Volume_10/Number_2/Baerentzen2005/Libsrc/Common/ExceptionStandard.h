#ifndef EXCEPTIONSTANDARD_H
#define EXCEPTIONSTANDARD_H

#include <string>
#include <iostream>

namespace Common
{

	class MotherException
	{
		std::string str;
	public:
		MotherException(const std::string s)
		{
			str = s;
		}
  
		void print(std::ostream& os) const 
		{
			os << str << std::endl; 
		}
	};

	/***************************************************************************
	 * The macro below makes it easier to create new exception classes. The    *
	 * first argument should be the name of the new exception, and the second  *
	 * argument should be the name of the exception it is to be derived from.  *
	 *                                                                         *
	 * All exceptions should (ultimately) be derived from the MotherException  *
	 * class.                                                                  *
	 *                                                                         *
	 * All exceptions should have a constructor which is passed a std::string  *
	 * describing the nature of the problem (why the exception is thrown)      *
	 *                                                                         *
	 * Just by using macro below, the last requirement is fulfilled. The first *
	 * one is fulfilled if the macro is used only with MotherException as      *
	 * `nameope' or with a class that is (directly or indirectly) derived      *
	 * from her.                                                               *
	 *                                                                         *
	 ***************************************************************************/

#define DERIVEEXCEPTION(nameoe,nameope)\
class nameoe: public nameope\
{\
public:\
 nameoe(const std::string& s): nameope(s) {}\
}\

}

namespace CMN = Common;


#endif
