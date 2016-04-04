
   
   An Image Inpainting Technique based on the Fast Marching Method
   ===============================================================
   A. C. Telea, Eindhoven University of Technology, 2003


   This is a C++ sample implementation of the inpainting technique
   based on the Fast Marching Method (FMM). The C++ sources can be
   compiled with Microsoft's Visual Studio. Alternatively, a simple
   makefile for other platforms can be easily created, if needed.

   The executable expects two file names for two images in 24-bit BMP format:

   - the 'original image', i.e. the image to inpaint
   - the 'scratch image', i.e. the image containing the region to inpaint
     marked with black pixels, and the known image area marked with white pixels

   Only the original image pixels corresponding to white scratch image pixels
   will be used. The original image pixels corresponding to black scratch
   image pixels can be anything, they will not be used during inpainting.
   The two images must have exactly the same dimension.


  

