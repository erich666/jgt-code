Demo code implementing Tiled Shading, accompanying the article 'Tiled Shading'
published in the journal of graphics, gpu, and game tools, in 2011.


BibTex entry:
---------------
@article{OlssonAssarsson2011,
  author = {Ola Olsson and Ulf Assarsson},
  title = {Tiled Shading},
  journal = {journal of graphics, gpu, and game tools},
  volume = {15},
  number = {4},
  pages = {235–-251},
  year = {2011},
  abstract = {In this article we describe and investigate tiled shading. The tiled techniques, though simple, enable substantial improvements to both deferred and forward shading. Tiled Shading has been previously discussed only in terms of deferred shading (tiled deferred shading). We contribute a more detailed description of the technique, introduce tiled forward shading (a generalization of tiled deferred shading to also apply to forward shading), and a thorough performance evaluation.

Tiled Forward Shading has many of the advantages of deferred shading, for example, scene management and light management are decoupled. At the same time, unlike traditional deferred and tiled deferred shading, full screen antialiasing and transparency are trivially supported.

We also present a thorough comparison of the performance of tiled deferred, tiled forward, and traditional deferred shading. Our evaluation shows that tiled deferred shading has the least variable worst-case performance, and scales the best with faster GPUs. Tiled deferred shading is especially suitable when there are many light sources. Tiled forward shading is shown to be competitive for scenes with fewer lights, and is much simpler than traditional forward shading techniques.

Tiled shading also enables simple transitioning between deferred and forward shading. We demonstrate how this can be used to handle transparent geometry, frequently a problem when using deferred shading.

Demo source code is available online at the address provided at the end of this paper.},
}



Usage Instructions
---------------------
Run the executable by starting the file 'run.cmd'. If this fails, the most
probable reason is that the system doesn't have the Visual Studio 
redistributable package(s) installed. Run 'install_redist.cmd' to install
the redistributables for both Visual C++ 2008 and 2005, this is required as
the DevIL binaries are linked against the 2005 version.

By default the demo will load the scene 'data/crysponza/sponza.obj', which is a
slightly modified and repackaged version of the scene made available by Crytek:
http://www.crytek.com/cryengine/cryengine3/downloads

To use another scene, just replace the command line argument. The only 
supported scene format is obj.

When the demo is running, pressing <F1> brings up an info/help text which 
provides details on other function keys. The help text also provides some
stats, such as number of lights used.

To change the maximum number of lights supported and other compile time 
options, see Config.h.



Programmer Guide
------------------
The most interesting files ought to be:

'tiled_deferred_demo.cpp' - contains the main program logic, and is admittedly
a bit of a monster. The rendering is controlled from the function onGlutDisplay,
so why not start there?

'shaders/tiledShading.glsl' - in which all the logic and uniforms needed to 
compute tiled shading, in the shaders, is contained. This file is included from
'tiled_forward_fragment.glsl' and 'tiled_deferred_fragment.glsl', and there
used to compute shading. 

'LightGrid.h/cpp' - contains the logic needed to construct the light grid on the
CPU.

'Config.h' - as noted before, this is where some of the core program behaviour
can be configured. For example, maximum number of lights and grid resolution.
Note that these properties may be subject to hardware/API restrictions, read
associated comments carefully.



Buiding the source
--------------------
This should be as easy as opening the solution and pressing whatever button 
does the build for you. Assuming, of course, that you have Visual Studio 2008
installed. Visual Studio 2005 has also been tested, but for the 2010 version
no project files are provided. There are are binaries and libraries for 32-bit
and 64-bit targets.

No other OS or environment is directly supported. However, most of the source
is standard C++ and some exist with a linux port, so a conversion should be 
relatively straight forward. The libraries that the demo depends on must be 
acquired independently. These are: DevIL, freeglut, and glew. 



System Requirements:
-----------------------
Windows XP or above
Graphics Card + Driver supporting OpenGL 3.3

