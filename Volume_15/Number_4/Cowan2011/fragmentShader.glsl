// A Real-Time, GPU-Based Method to Approximate Acoustical Reverberation Effects
// Brent Cowan and Bill Kapralos

// Faculty of Business and Information Technology, Health Education Technology Research Unit.
// University of Ontario Institute of Technology.
//
// January 25 2010

// Fragment Shader

uniform vec4 properties; //surface properties
varying vec3 normal;
varying vec3 position;

void main ()
{
	vec3 pos = normalize(position);
	vec3 norm = normalize(normal);
	float angle = dot(pos, norm) * properties.x;

	float dist = length(position);
	dist = dist/25.0;

	gl_FragColor = vec4(dist, properties.x, angle, 1.0);
}