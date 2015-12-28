// bdl 13. nov 2001
#pragma warning (disable : 4786) 

#include <stdio.h>
#include "DisplayText.h"
#include <GL/glut.h>
#include "Components/Timer.h"

using namespace std;

namespace Graphics
{

	DisplayText::DisplayText() {
		id = 0;
		x_cur = 0;
		y_cur = 0;
		x_offset = 0;
		y_offset = -14;
		framerate_id = -1;
	}

	int DisplayText::setText(std::string text, int x, int y) {
		TextObject *textobject = new TextObject();
		textobject->text = text;
		x_cur = x;
		y_cur = y;
		textobject->x = x;
		textobject->y = y;
		textobjects[id] = textobject;
		return id++;
	}

	int DisplayText::addText(std::string text) {
		x_cur += x_offset;
		y_cur += y_offset;
		return setText(text, x_cur, y_cur);
	}

	void DisplayText::updateText(int id, std::string text) {
		textobjects[id]->text = text;
	}

	void DisplayText::setOffset(int _x_offset, int _y_offset) {
		x_offset = _x_offset;
		y_offset = _y_offset;
	}

	void DisplayText::addFramerate() {
		framerate_id = addText("fps:??");
	}

	void DisplayText::oneFrame() {
		if (framerate_id==-1)
			return;
		static int i=0;
		const int interval = 30;
		static CMP::Timer stopwatch;
		if (i>=interval) {
			//stopwatch.stop_timer();
			float time_counter = stopwatch.get_secs();
			float fps = static_cast<float>(interval)/(time_counter);
			char str[60];
			sprintf(str,"fps:%4.1f",fps);
			string s = string(str);
			textobjects[framerate_id]->text = s;
			i=0;
		}
		if (i==0)
			stopwatch.start();
		i++;
	}

	void DisplayText::draw() {
		glPushAttrib(GL_ENABLE_BIT);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);

		oneFrame();
		int ww = glutGet( (GLenum)GLUT_WINDOW_WIDTH );
		int wh = glutGet( (GLenum)GLUT_WINDOW_HEIGHT );

		glMatrixMode( GL_PROJECTION );
		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D( 0, ww-1, 0, wh-1 );
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadIdentity();
		glColor3f(1,0,0);
		map<int,TextObject*>::iterator cur; 
		for(cur=textobjects.begin();cur!=textobjects.end();++cur) {
			int x = 20 + (*cur).second->x;
			int y = wh-22 + (*cur).second->y;

			glRasterPos2i( x, y );
			char string[1024];
			strcpy(string,(*cur).second->text.c_str());
		
			char *p;

			for ( p = string; *p; p++ )
				{
					if ( *p == '\n' )
						{
							y = y - 14;
							glRasterPos2i( x, y );
							continue;
						}
					glutBitmapCharacter( GLUT_BITMAP_9_BY_15, *p );
				}
		}

		glMatrixMode( GL_PROJECTION );
		glPopMatrix();
		glMatrixMode( GL_MODELVIEW );
		glPopMatrix();
		glPopAttrib();
	}

}
