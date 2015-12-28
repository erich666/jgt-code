#ifndef ARG_EXTRACTER
#define ARG_EXTRACTER
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <list>
#include <string>
#include "ExceptionStandard.h"

namespace Common
{

	DERIVEEXCEPTION(JabStringException, MotherException);

	template<class T>
	T string_convert(const std::string& x)
	{
		throw JabStringException("Illegal to use un-specialized std::string_convert");
		T t;
		return t;
	}
	template<> int string_convert(const std::string& x){ 
		return std::atoi(x.c_str());}
	template<> float string_convert(const std::string& x){ 
		return std::atof(x.c_str());}
	template<> std::string string_convert(const std::string& x){ 
		return x;}


	struct UpCase {void operator()(char& x) {x=toupper(x);}};

	void up_case_string(std::string& s)
	{
		std::for_each(s.begin(), s.end(), UpCase());
	}




	class ArgExtracter
	{	
		std::list<std::string> avec;
		typedef std::list<std::string>::iterator LSI;

		bool extract(const std::string& argname, LSI& iter)
		{
			for(iter = avec.begin();iter != avec.end(); ++iter)
				{
					if((*iter)==argname)
						{
							iter = avec.erase(iter);
							return true;
						}
				}
			return false;
		}

	public:

		ArgExtracter(int argc, char **argv)
		{
			for(int i=0;i<argc; ++i)
				avec.push_back(std::string(argv[i]));
		}

		bool extract(const std::string& argname)
		{
			LSI iter;
			return extract(argname, iter);
		}

		template<class T>
		bool extract(const std::string& argname, T& val)
		{
			LSI iter;
			if(extract(argname, iter))
				{
					val = string_convert<T>(iter->c_str());
					avec.erase(iter);
					return true;
				}
			return false;
		}

		int no_remaining_args() const
		{
			return avec.size();
		}

		const std::string& get_last_arg() const
		{
			return avec.back();
		}

		void get_all_args(std::vector<std::string>& args)
			{
				LSI iter;
				args = std::vector<std::string>(avec.begin(), avec.end());
			}
	};

}

namespace CMN = Common;

#endif
