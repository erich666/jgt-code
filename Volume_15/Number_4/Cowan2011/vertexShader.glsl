// A Real-Time, GPU-Based Method to Approximate Acoustical Reverberation Effects
// Brent Cowan and Bill Kapralos

// Faculty of Business and Information Technology, Health Education Technology Research Unit.
// University of Ontario Institute of Technology.
//
// January 25 2010

// Vertex Shader

varying vec3 normal;
varying vec3 position;

void main ()
{
	gl_Position = ftransform();
	normal = normalize(gl_NormalMatrix * gl_Normal);


	position = -vec3(gl_ModelViewMatrix * gl_Vertex);
}