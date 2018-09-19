// A Real-Time, GPU-Based Method to Approximate Acoustical Reverberation Effects
// Brent Cowan and Bill Kapralos

// Faculty of Business and Information Technology, Health Education Technology Research Unit.
// University of Ontario Institute of Technology.
//
// January 25 2010

// CPU Code: code to process the output from the shader on the CPU. 
	
glReadPixels(x, y, width, height, GL_RGB,  GL_FLOAT, output);

for(GLuint i=0; i<256; i++)
{
	data[i] = 0.0f;
}

roomSize = 0.0f;
reflection = 0.0f;
	
//Process the image into a distogram
for(GLuint i=0; i<width*height*3; i+=3)
{
	reflection += output[i+2];
	data[GLushort(output[i]*255.0f)] += output[i+2];
}
	
//Make the output in a range between 0 and 1
reflection = reflection/float(width*height);

//Not necessary, but makes reverberation more pronounced for distant surfaces.
//Some 1D filtering can be done here to adjust the output to suit a variety of sound engines. 
reflection = 1.0f-reflection;
reflection = reflection*reflection;
reflection = 1.0f-reflection;

if(reflection>1.0f)reflection=1.0f;//cap the range

float weight = 0.0f;
for(GLuint i=0; i<256; i++)
{
	roomSize += (float(i)/255.0f) * data[i];
	weight += data[i];
}
roomSize = roomSize/weight;