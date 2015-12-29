This demo provides source for the paper ``Reducing Noise in Image-Space
Caustics with Variable-Size Splatting.''  The executable in the base
directory should work as-is, given the following requirements are met.

Requirements:
-------------
   * Windows
   * OpenGL 2.0 compatible drivers
   * Cg installed (with .DLLs in the path)
   * Highly likely that a nVidia card is necessary 
        (I developed on a 6800, 7800, 7800 Go)
   * GLUT .DLL installed and in the path (http://www.xmission.com/~nate/glut.html)
   * GLEW .DLL installed and in the path (http://glew.sourceforge.net)

Program Execution:
------------------

Simply double clicking will load a default scene (with the Beethovan bust), though
a number of ``.settings'' files are included which can be passed in as a command-line
parameter to load various settings (e.g., ``improvedCaustics.exe DragonOnDragon.settings'').
Most relevant settings can be controlled via the GLUT menu (right click), and a variety
of other settings can be keystrokes (press 'h' for a short list of useful keys).


Where to look for relevent C++ code:  (relatively well commented)
-----------------------------------------------------------------
   improvedCaustics.cpp           <--- main() and display()
   basicRefractionFunctions.cpp   <--- code for doing image-space refraction
   lightSpacePhotonGather.cpp     <--- code for performing the 3 types of caustic rendering


Where to look for relevent Cg shaders:  
------------------------------------------------------------------------------------------------
   shaders/refractionWBackground/*           <--- original refraction & caustics mapping code
   shaders/storePhotonsInLightBuffer*.cg     <--- original caustic mapping code
   shaders/lightGatherWGaussianSplats.*.cg   <--- original caustic mapping code

   shaders/compineMultiLevelGaussianSplats.*.cg    <--- code for combining multi resolution splats together

   * Modified shaders from the original caustic mapping code include:  
        refractionOtherObjsFrag.justPhotonData.cg  <--- added thin/thick lens approximation for splat sizes
        lightGatherWGaussianSplats.Vert.cg         <--- added dynamically changing point size (based on lens value)

