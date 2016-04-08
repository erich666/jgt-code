#include "data_path.h"
#include <stdio.h>
#include <io.h>
#include <fcntl.h>

using namespace std;


// data files, for read only
FILE * data_path::fopen(std::string filename, const char * mode)
{

	for(int i=0; i < path.size(); i++)
	{
		std::string s = path[i] + "/" + filename;
		FILE * fp = ::fopen(s.c_str(), mode);

		if(fp != 0)
			return fp;
	}
	// no luck... return null
	return 0;
}

//  fill the file stats structure 
//  useful to get the file size and stuff
int data_path::fstat(std::string filename, struct _stat * stat)
{
	for(int i=0; i < path.size(); i++)
	{
		std::string s = path[i] + "/" + filename;
		int fh = ::_open(s.c_str(), _O_RDONLY);

		if(fh != -1)
        {
            int result = ::_fstat( fh, stat );
            if( result != 0 )
            {
                fprintf( stderr, "An fstat error occurred.\n" );
                return 0;
            }
            ::_close( fh );
			return 1;
    	}
    }
    // no luck...
    return 0;
}