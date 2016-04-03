#define RETURN_HWB(h, w, b) {HWB.H = h; HWB.W = w; HWB.B = b; return HWB;}

#define RETURN_RGB(r, g, b) {RGB.R = r; RGB.G = g; RGB.B = b; return RGB;}

#define UNDEFINED -1

// Theoretically, hue 0 (pure red) is identical to hue 6 in these transforms. Pure

// red always maps to 6 in this implementation. Therefore UNDEFINED can be

// defined as 0 in situations where only unsigned numbers are desired.

typedef struct {float R, G, B;} RGBType;

typedef struct {float H, W, B;} HWBType;

HWBType

RGB_to_HWB( RGBType RGB ) {

	// RGB are each on [0, 1]. W and B are returned on [0, 1] and H is
	// returned on [0, 6]. Exception: H is returned UNDEFINED if W == 1 - B.
	float R = RGB.R, G = RGB.G, B = RGB.B, w, v, b, f;
	int i;
	HWBType HWB;
	w = min(R, G, B);
	v = max(R, G, B);
	b = 1 - v;
	if (v == w) RETURN_HWB(UNDEFINED, w, b);
	f = (R == w) ? G - B : ((G == w) ? B - R : R - G);
	i = (R == w) ? 3 : ((G == w) ? 5 : 1);
	RETURN_HWB(i - f /(v - w), w, b);
}

RGBType

HWB_to_RGB( HWBType HWB ) {

	// H is given on [0, 6] or UNDEFINED. W and B are given on [0, 1].
	// RGB are each returned on [0, 1].
	float h = HWB.H, w = HWB.W, b = HWB.B, v, n, f;
	int i;
	RGBType RGB;
	v = 1 - b;
	if (h == UNDEFINED) RETURN_RGB(v, v, v);
	i = floor(h);
	f = h - i;
	if (i & 1) f = 1 - f; // if i is odd
	n = w + f * (v - w); // linear interpolation between w and v
	switch (i) {
		case 6:
		case 0: RETURN_RGB(v, n, w);
		case 1: RETURN_RGB(n, v, w);
		case 2: RETURN_RGB(w, v, n);
		case 3: RETURN_RGB(w, n, v);
		case 4: RETURN_RGB(n, w, v);
		case 5: RETURN_RGB(v, w, n);
	}
}