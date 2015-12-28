#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif


#include "PointRecord.h"

using namespace std;

bool save_points(const std::string& fname, 
								 const std::vector<PointRecord>& points,
								 float unit_scale)
{
	int f = creat(fname.data(), 0644);

	if(f != -1)
		{
			char magic[] = "POINTS";
			if(write(f,magic,7) == -1) 
				return false;
			
			int N = points.size();
			if(write(f, reinterpret_cast<char*>(&N), sizeof(int)) == -1) 
				return false;

			if(write(f, reinterpret_cast<char*>(&unit_scale), sizeof(float)) == -1) 
				return false;

			int sz = sizeof(PointRecord)* N; 
			if(write(f, reinterpret_cast<const char*>(&points[0]), sz) == -1) 
				return false;

			close(f);
			return true;
		}
	return false;
}

bool load_points(const std::string& fname, 
								 std::vector<PointRecord>& points,
								 float& unit_scale)
{
	int f = open(fname.data(), O_RDONLY);

	if(f != -1)
		{
			char magic[7];
			if(read(f,magic,7) == -1) 
				return false;
			if(string(magic) != string("POINTS"))
				return false;
			
			int N;
			if(read(f, reinterpret_cast<char*>(&N), sizeof(int)) == -1) 
				return false;
			points.resize(N);

			if(read(f, reinterpret_cast<char*>(&unit_scale), sizeof(float)) == -1) 
				return false;
			
			int sz = sizeof(PointRecord)* N; 
			if(read(f, reinterpret_cast<char*>(&points[0]), sz) == -1) 
				return false;

			close(f);
			return true;
		}
	return false;
}


