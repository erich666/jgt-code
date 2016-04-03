#define RETURN_HSV(h, w, b) {HSV.H = h; HSV.S = s; HSV.V = v; return HSV;}

#define RETURN_RGB(r, g, b) {RGB.R = r; RGB.G = g; RGB.B = b; return RGB;}

#define UNDEFINED -1

// Theoretically, hue 0 (pure red) is identical to hue 6 in these transforms. Pure

// red always maps to 6 in this implementation. Therefore UNDEFINED can be

// defined as 0 in situations where only unsigned numbers are desired.

typedef struct {float R, G, B;} RGBType;

typedef struct {float H, S, V;} HSVType;

// HSVType

RGB_to_HSV( RGBType RGB ) {

	// RGB are each on [0, 1]. S and V are returned on [0, 1] and H is
	// returned on [0, 6]. Exception: H is returned UNDEFINED if S==0.
	float R = RGB.R, G = RGB.G, B = RGB.B, v, x, f;
	int i;
	HSVType HSV;
	x = min(R, G, B);
	v = max(R, G, B);
	if(v == x) RETURN_HSV(UNDEFINED, 0, v);
	f = (R == x) ? G - B : ((G == x) ? B - R : R - G);
	i = (R == x) ? 3 : ((G == x) ? 5 : 1);
	RETURN_HSV(i - f /(v - x), (v - x)/v, v);
}

// RGBType

HSV_to_RGB( HSVType HSV ) {

	// H is given on [0, 6] or UNDEFINED. S and V are given on [0, 1].
	// RGB are each returned on [0, 1].
	float h = HSV.H, s = HSV.S, v = HSV.V, m, n, f;
	int i;
	RGBType RGB;
	if(h == UNDEFINED) RETURN_RGB(v, v, v);
	i = floor(h);
	f = h - i;
	if(!(i & 1)) f = 1 - f; // if i is even
	m = v * (1 - s);
	n = v * (1 - s * f);
	switch (i) {
		case 6:
		case 0: RETURN_RGB(v, n, m);
		case 1: RETURN_RGB(n, v, m);
		case 2: RETURN_RGB(m, v, n)
		case 3: RETURN_RGB(m, n, v);
		case 4: RETURN_RGB(n, m, v);
		case 5: RETURN_RGB(v, m, n);
	}
}