#include <math.h>

#define  DEG2RAD   0.01745329252f
inline float fdegsin(float v) { return (float) sin((double) (v * DEG2RAD)); }
inline float fdegcos(float v) { return (float) cos((double) (v * DEG2RAD)); }

inline void getIdentityMat(float m[16]) {
	m[0] = 1.0f; m[4] = 0.0f; m[8]  = 0.0f; m[12] = 0.0f; 
	m[1] = 0.0f; m[5] = 1.0f; m[9]  = 0.0f; m[13] = 0.0f; 
	m[2] = 0.0f; m[6] = 0.0f; m[10] = 1.0f; m[14] = 0.0f; 
	m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f; 
}

inline void getRotX(float a, float m[16]) {
	float sina = fdegsin(a);
	float cosa = fdegcos(a);
	m[0] = 1.0f; m[4] = 0.0f; m[8]  =  0.0f; m[12] = 0.0f; 
	m[1] = 0.0f; m[5] = cosa; m[9]  = -sina; m[13] = 0.0f; 
	m[2] = 0.0f; m[6] = sina; m[10] =  cosa; m[14] = 0.0f; 
	m[3] = 0.0f; m[7] = 0.0f; m[11] =  0.0f; m[15] = 1.0f; 
}

inline void getRotY(float a, float m[16]) {
	float sina = fdegsin(a);
	float cosa = fdegcos(a);
	m[0] =  cosa; m[4] = 0.0f; m[8]  = sina; m[12] = 0.0f; 
	m[1] =  0.0f; m[5] = 1.0f; m[9]  = 0.0f; m[13] = 0.0f; 
	m[2] = -sina; m[6] = 0.0f; m[10] = cosa; m[14] = 0.0f; 
	m[3] =  0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f; 
}

inline void getRotZ(float a, float m[16]) {
	float sina = fdegsin(a);
	float cosa = fdegcos(a);
	m[0] = cosa; m[4] = -sina; m[8]  = 0.0f; m[12] = 0.0f; 
	m[1] = sina; m[5] =  cosa; m[9]  = 0.0f; m[13] = 0.0f; 
	m[2] = 0.0f; m[6] =  0.0f; m[10] = 1.0f; m[14] = 0.0f; 
	m[3] = 0.0f; m[7] =  0.0f; m[11] = 0.0f; m[15] = 1.0f; 	
}

inline void mulMatMat(float m[16], float n[16], float r[16]) {
	float t[16];
	
	t[0]  = m[0] * n[0]  + m[4] * n[1]  + m[8] * n[2]  + m[12] * n[3];
	t[4]  = m[0] * n[4]  + m[4] * n[5]  + m[8] * n[6]  + m[12] * n[7];
	t[8]  = m[0] * n[8]  + m[4] * n[9]  + m[8] * n[10] + m[12] * n[11];
	t[12] = m[0] * n[12] + m[4] * n[13] + m[8] * n[14] + m[12] * n[15];

	t[1]  = m[1] * n[0]  + m[5] * n[1]  + m[9] * n[2]  + m[13] * n[3];
	t[5]  = m[1] * n[4]  + m[5] * n[5]  + m[9] * n[6]  + m[13] * n[7];
	t[9]  = m[1] * n[8]  + m[5] * n[9]  + m[9] * n[10] + m[13] * n[11];
	t[13] = m[1] * n[12] + m[5] * n[13] + m[9] * n[14] + m[13] * n[15];

	t[2]  = m[2] * n[0]  + m[6] * n[1]  + m[10] * n[2]  + m[14] * n[3];
	t[6]  = m[2] * n[4]  + m[6] * n[5]  + m[10] * n[6]  + m[14] * n[7];
	t[10] = m[2] * n[8]  + m[6] * n[9]  + m[10] * n[10] + m[14] * n[11];
	t[14] = m[2] * n[12] + m[6] * n[13] + m[10] * n[14] + m[14] * n[15];

	t[3]  = m[3] * n[0]  + m[7] * n[1]  + m[11] * n[2]  + m[15] * n[3];
	t[7]  = m[3] * n[4]  + m[7] * n[5]  + m[11] * n[6]  + m[15] * n[7];
	t[11] = m[3] * n[8]  + m[7] * n[9]  + m[11] * n[10] + m[15] * n[11];
	t[15] = m[3] * n[12] + m[7] * n[13] + m[11] * n[14] + m[15] * n[15];

	for (int i = 0; i < 15; i++)
		r[i] = t[i];
}

inline void getEulerRotation(float ax, float ay, float az, float m[16]) {	
	float mx[16], my[16], mz[16];

	getRotX(ax, mx);
	getRotY(ay, my);
	getRotZ(az, mz);

	mulMatMat(my, mz, m);
	mulMatMat(mx, m, m);
}

inline void mulMatVec(float m[16], float p[3], float r[3]) {
	float t[3];

	t[0] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
	t[1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
	t[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];

	r[0] = t[0];
	r[1] = t[1];
	r[2] = t[2];
}
