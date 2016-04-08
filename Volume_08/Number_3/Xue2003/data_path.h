#ifndef DATA_PATH_H
#define DATA_PATH_H

#pragma warning(disable:4786)   // symbol size limitation ... STL


#include <stdio.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

class data_path
{
public:

	std::vector<std::string> path;

	// data files, for read only
	FILE *  fopen(std::string filename, const char * mode = "rb");
    int     fstat(std::string filename, struct _stat * stat);

};

#endif
