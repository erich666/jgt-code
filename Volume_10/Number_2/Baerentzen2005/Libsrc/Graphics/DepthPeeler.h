#ifndef __LDI_GENERATION_H
#define __LDI_GENERATION_H

namespace Graphics
{
	class DepthPeeler
	{
		unsigned int frag_prog, ztex, width, height;
		double texmat[16];
	public:

		DepthPeeler(int width, int height);
		~DepthPeeler();


		void DepthPeeler::disable_depth_test2();
		void DepthPeeler::enable_depth_test2();

		void read_back_depth();
	};
}
namespace GFX = Graphics;
#endif
