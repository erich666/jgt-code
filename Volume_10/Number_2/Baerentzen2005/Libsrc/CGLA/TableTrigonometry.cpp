#include "TableTrigonometry.h"

namespace CGLA {

	using namespace std;

	namespace TableTrigonometry
	{
		const CosTable& COS_TABLE()
		{
			static CosTable table;
			return table;
		}
	}

}
