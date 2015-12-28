#ifndef __DISPLAYTEXT_H__
#define __DISPLAYTEXT_H__

#include <string>
#include <map>

namespace Graphics
{

	struct TextObject {
		int x;
		int y;
		int id;
		std::string text;
	};

	class DisplayText {
		int id;
		int x_cur;
		int y_cur;
		int x_offset;
		int y_offset;
		std::map<int, TextObject*> textobjects;
		int framerate_id;
		void oneFrame();
	public:
		DisplayText();

		int setText(std::string text, int x, int y);
		void updateText(int id, std::string text);
		void setOffset(int x_offset, int y_offset);
	
		int addText(std::string text);
		void addFramerate();

		void draw();	

	};
}
namespace GFX=Graphics;
#endif
