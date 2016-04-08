#include "read_text_file.h"
#include "data_path.h"
#include <stdio.h>

using namespace std;

namespace
{
	data_path path;
}

data_path get_text_path() { return path; }
void      set_text_path(const data_path & newpath) { path = newpath; }

char * read_text_file(const char * filename)
{
	if(path.path.size() < 1)
	{
		path.path.push_back(".");
		path.path.push_back("./data/programs");
		path.path.push_back("./data/programs");
	}

    if (!filename) return 0;

    struct _stat stat;
    if (!path.fstat(filename, &stat))
	{
		fprintf(stderr,"Cannot open \"%s\" for stat read!\n", filename);
		return 0;
	}
    long size = stat.st_size;

    char * buf = new char[size+1];

	FILE *fp = 0;
    if (!(fp = path.fopen(filename, "r")))
	{
		fprintf(stderr,"Cannot open \"%s\" for read!\n", filename);
		return 0;
	}

	int bytes;
	bytes = fread(buf, 1, size, fp);

    buf[bytes] = 0;

	fclose(fp);
	return buf;
}