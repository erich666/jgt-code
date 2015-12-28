
#include <FL/GL.H>
#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>

#include "JofGTIBar.h"
#include "UserInterface.h"

double RNEpsilon_d = 1e-16;

int main( int argc, char ** argv )
{
    // Create and run Window
    UserInterface *window = new UserInterface( );
  
    window->m_mainWindow->show();

    return Fl::run();
}
