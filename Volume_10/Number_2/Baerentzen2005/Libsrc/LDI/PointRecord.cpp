#include <fstream>
#include "PointRecord.h"

namespace LDI
{
	const int K=125;
	using namespace std;

	bool save_points(const std::string& fname, 
									 const std::vector<PointRecord>& points,
									 float unit_scale)
	{
		ofstream f(fname.data(),ios::binary|ios::trunc);
		if(f.good())
			{
				char magic[] = "POINTS";
				f.write(magic,7);
							
				int N = points.size();
				f.write(reinterpret_cast<char*>(&N), sizeof(int));

				f.write(reinterpret_cast<char*>(&unit_scale), sizeof(float)); 

				int sz = sizeof(PointRecord)* N; 
				f.write(reinterpret_cast<const char*>(&points[0]), sz);

				return true;
			}
		return false;
	}

	bool load_points(const std::string& fname, 
									 std::vector<PointRecord>& points,
									 float& unit_scale)
	{
		ifstream f(fname.data(),ios::binary);
		if(f.good())
			{
				char magic[7];
				if(f.read(magic,7));
					
				if(string(magic) != string("POINTS"))
					return false;
			
				int N;
				f.read(reinterpret_cast<char*>(&N), sizeof(int));
				points.resize(N);

				f.read(reinterpret_cast<char*>(&unit_scale), sizeof(float));
		
				int sz = sizeof(PointRecord)* N; 
				f.read(reinterpret_cast<char*>(&points[0]), sz);

				return true;
			}
		return false;
	}

}
