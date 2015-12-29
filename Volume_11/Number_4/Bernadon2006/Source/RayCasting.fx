//-----------------------------------------------------------------------------
// File: RayCasting.fx
// Desc: shader programs for hardware ray casting
// Copyright (C) 2005, Joao Comba, Fabio Bernardon, UFRGS-Brasil
//-----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Input parameters 
//-----------------------------------------------------------------------------

#include "../Header/defines.h"

//new constants - old assembler codes will not work!
float4x4 matModel : register(c0);
float4x4 matView : register(c4);
float4x4 matProjection : register(c8);
float4x4 matModelInverse : register(c12);
float4x4 matViewInverse;
float4x4 matViewInverseTransposed;
float4x4 matViewInverseTransposedInverse;
float4x4 matProjectionInverse;
float4x4 matProjectionInverseTransposedInverse;

//hold the pass color - old vectors kept for asm code compatibility 
float4 lrpParam : register(c26);
float4 passColor : register(c28);
float4 colorScaleBias : register(c30);
float4 eyeLocal : register(c31);

float nPass;
float textureDimension;
float textureDimensionDiv3;
float normalizedTexDimDiv3;

TEXTURE tCurrentCell;
TEXTURE tColor;
TEXTURE tLut;
TEXTURE tLut2D;

TEXTURE tNewVertex;
TEXTURE tNewNormals;
TEXTURE tNewNeighbor;

//create the samples related whit the textures

sampler newVertexSampler = sampler_state
{
Texture   = <tNewVertex>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler newNormalsSampler = sampler_state
{
Texture   = <tNewNormals>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler newNeighborSampler = sampler_state
{
Texture   = <tNewNeighbor>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler currentCellSampler = sampler_state
{
Texture   = <tCurrentCell>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler colorSampler = sampler_state
{
Texture   = <tColor>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler lutSampler = sampler_state
{
Texture   = <tLut>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};

sampler lut2DSampler = sampler_state
{
Texture   = <tLut2D>;
MinFilter = POINT;
MagFilter = POINT;
MipFilter = POINT;
};


//-----------------------------------------------------------------------------
// Vertex Shader INPUTS/OUTPUTS
//-----------------------------------------------------------------------------
struct Test_VS_INPUT
{
	float4 position		      : POSITION;
	float4 color		        : COLOR;
	float4 texCoord0	      : TEXCOORD0;
	float4 texCoord1	      : TEXCOORD1;
	float4 texCoord2	      : TEXCOORD2;
};

struct Test_VS_OUTPUT
{
	float4 position		      : POSITION;
	float4 color		        : COLOR;
	float4 texCoord0	      : TEXCOORD0;
	float4 texCoord1	      : TEXCOORD1;
	float4 texCoord2	      : TEXCOORD2;
};

struct RayCasting_VS_INPUT
{
	float4 position         : POSITION;
	float4 diffuse	        : COLOR;
	float4 texCoord0        : TEXCOORD0;
	float4 texCoord1	      : TEXCOORD1;
	float4 texCoord2	      : TEXCOORD2;
};

struct RayCasting_VS_OUTPUT
{
    float4 position		    : POSITION;
    float4 texCoord0	    : TEXCOORD0;
    float4 texCoord1	    : TEXCOORD1;
  	float4 tEye			      : TEXCOORD2;

};

struct FirstPass_VS_INPUT
{
	float4 position			    : POSITION;
	float4 texcoord0		    : NORMAL;
};

struct FirstPass_VS_OUTPUT
{
    float4 position			  : POSITION;
    float4 texcoord0		  : TEXCOORD0;
    float4 texcoord1	    : TEXCOORD1;
    float4 texcoord2		  : TEXCOORD2;
    float4 ray				    : TEXCOORD3;
};

//-----------------------------------------------------------------------------
// Pixel Shader INPUTS/OUTPUTS
//-----------------------------------------------------------------------------
struct Test_PS_INPUT
{
	float4 color		      : COLOR;
	float4 texCoord0	    : TEXCOORD0;
	float4 texCoord1	    : TEXCOORD1;
	float4 texCoord2	    : TEXCOORD2;
};

struct Test_PS_OUTPUT
{
	float4 color		      :COLOR;
};

struct RayCasting_PS_INPUT
{
	float4 rasterPos	    : TEXCOORD0;
	float4 ray			      : TEXCOORD1;
	float4 tEye			      : TEXCOORD2;
};

struct RayCasting_PS_OUTPUT
{
    float4 currentCell  : COLOR;
    float4 color        : COLOR1;
};

struct FirstPass_PS_INPUT
{
    float4 texcoord0	  : TEXCOORD0;
    float4 texcoord1	  : TEXCOORD1;
    float4 texcoord2	  : TEXCOORD2;
    float4 ray			    : TEXCOORD3;
};

struct FirstPass_PS_OUTPUT
{
    float4 currentCell  : COLOR;
};

struct DepthPeeling_PS_OUTPUT
{
    float4 currentCell  : COLOR;
    float depth			    : DEPTH;
};

struct Single_PS_OUTPUT
{
    float4 color        : COLOR;
};

struct Depth_PS_INPUT
{
    float4 diffuse		  : COLOR;
	float4 rasterPos	    : TEXCOORD0;
	float4 ray			      : TEXCOORD1;
	float4 tEye			      : TEXCOORD2;
};

struct Depth_PS_OUTPUT
{
	float4 color			    : COLOR;
	float depth				    : DEPTH;
};

struct IntersectColor_PS_OUTPUT
{
    float4 intersectPt  : COLOR;
    float4 color        : COLOR1;
};


//-----------------------------------------------------------------------------
// Old First hit shader
//-----------------------------------------------------------------------------
FirstPass_PS_OUTPUT FirstHitOld_PS(FirstPass_PS_INPUT IN)
{
	FirstPass_PS_OUTPUT OUT = (FirstPass_PS_OUTPUT) 0;

	//save the tetrahedral index
	OUT.currentCell = float4(IN.texcoord0.rgb, IN.texcoord2.z/IN.texcoord2.w/*, 1.0*/); 
	return OUT;
}

//-----------------------------------------------------------------------------
// Old Depth peeling pixel shader
//-----------------------------------------------------------------------------
DepthPeeling_PS_OUTPUT DepthPeelingOld_PS(FirstPass_PS_INPUT IN)
{
	DepthPeeling_PS_OUTPUT OUT = (DepthPeeling_PS_OUTPUT) 0;

	float2 index = (IN.texcoord2.xy/IN.texcoord2.w+1)/2 + 
					float2(0.0009765625, -0.0009765625);
	index.y = 1.0 - index.y;
	float4 currentValue = tex2D(currentCellSampler, index);
	float thisZ = IN.texcoord2.z / IN.texcoord2.w;

	//if newZ > currentZ + bias
	if (thisZ > (currentValue.w + 0.000004)){
		OUT.currentCell = float4(IN.texcoord0.rgb, thisZ/*, 1.0*/);
	OUT.currentCell = float4(IN.texcoord0.rgb, IN.texcoord2.z/IN.texcoord2.w/*, 1.0*/); 
		OUT.depth = thisZ;
	}else{
		//this value must be written behind all the others
		OUT.currentCell = float4(-1.0, 0.0, 1.0, 1.0);
		OUT.depth = 1.0f;
	}

	return OUT;
}


//#define USE_LAMBDA
#define NOVO_RC


//-----------------------------------------------------------------------------
// RenderDepthPeeling shader
//-----------------------------------------------------------------------------
Test_PS_OUTPUT RenderDepthPeeling_PS(Test_PS_INPUT IN)
{
	Test_PS_OUTPUT OUT = (Test_PS_OUTPUT) 0;

	float4 currentCell	= tex2D(currentCellSampler, IN.texCoord0.xy);

	OUT.color	= float4(0.0, 0.0, 0.0, 1.0);

	if(currentCell.x <= 0.0)
		OUT.color = float4(0.0, 0.0, 0.0, 1.0);
	else
		OUT.color = float4(0.0, 1.0, 0.0, 1.0);

	return OUT;
}


//-----------------------------------------------------------------------------
// FaceColored shader
//-----------------------------------------------------------------------------
Test_VS_OUTPUT FaceColored_VS(Test_VS_INPUT IN)
{
	Test_VS_OUTPUT OUT = (Test_VS_OUTPUT) 0;

    OUT.position  = IN.position;
	OUT.texCoord0 = IN.texCoord0;
	OUT.texCoord1 = IN.texCoord1;
	OUT.texCoord2 = IN.texCoord2;

    return OUT;
}

//-----------------------------------------------------------------------------
// FaceColored shader
//-----------------------------------------------------------------------------
Test_PS_OUTPUT FaceColored_PS(Test_PS_INPUT IN)
{
	Test_PS_OUTPUT OUT = (Test_PS_OUTPUT) 0;

	float4 color = float4(0.0, 0.0, 0.0, 0.0);

	float4 currentCell	= tex2D(currentCellSampler, IN.texCoord0.xy);

	OUT.color	= float4(1.0, 1.0, 1.0, 1.0);								

	if(currentCell.x <= 1.0 && currentCell.y <= 1.0 &&
			currentCell.x > 0.0 && currentCell.y >= 0.0)
		OUT.color = float4(currentCell.x, 1.0 - currentCell.y, 0.0, 1.0);
	else
		OUT.color = float4(0.0, 0.0, 0.0, 1.0);

	return OUT;
}

//-----------------------------------------------------------------------------
// ShowColorBuffer
//-----------------------------------------------------------------------------
Test_VS_OUTPUT ShowColorBuffer_VS(Test_VS_INPUT IN)
{
	Test_VS_OUTPUT OUT = (Test_VS_OUTPUT) 0;

    OUT.position  = IN.position;
	OUT.texCoord0 = IN.texCoord0;
	OUT.texCoord1 = IN.texCoord1;
	OUT.texCoord2 = IN.texCoord2;

    return OUT;
}

//-----------------------------------------------------------------------------
// ShowColorBuffer
//-----------------------------------------------------------------------------
Test_PS_OUTPUT ShowColorBuffer_PS(Test_PS_INPUT IN)
{
	Test_PS_OUTPUT OUT = (Test_PS_OUTPUT) 0;

	float4 color;

	color  = tex2D(colorSampler, float2(IN.texCoord0.xy));
	//color.w = 0;
	OUT.color = color;
//	OUT.color = float4(color.xyz * (1.0 - color.w), 1.0);

	return OUT;
}


//-----------------------------------------------------------------------------
// Pass Through pixel shader
//-----------------------------------------------------------------------------
Single_PS_OUTPUT PassThrough_PS(RayCasting_PS_INPUT IN)
{
	Single_PS_OUTPUT OUT = (Single_PS_OUTPUT) 0;

	OUT.color = float4(0.0, 0.0, 0.0, 0.0);

	return OUT;
}


//-----------------------------------------------------------------------------
// Depth pass for occlusion test
//-----------------------------------------------------------------------------
Depth_PS_OUTPUT Depth_PS(RayCasting_PS_INPUT IN)
{
	Depth_PS_OUTPUT OUT = (Depth_PS_OUTPUT) 0;

	// Retrieve data from traversal textures
	float4 currentCell	= tex2D(currentCellSampler, IN.rasterPos.xy);
	// Test if the ray already left the mesh
	OUT.depth = (currentCell.x <= 0.00001 /*|| 
				currentCell.z > 99999.0*/) ? 0.01f : 1.0f;
	OUT.color = (currentCell.x <= 0.0 || currentCell.z > 99999.0) ?float4(0.0, 0.0, 999999.0, 0.0) : currentCell;

	return OUT;
}


//-----------------------------------------------------------------------------
// Depth Thorugh pass for occlusion test
//-----------------------------------------------------------------------------
Depth_PS_OUTPUT DepthThrough_PS(RayCasting_PS_INPUT IN)
{
	Depth_PS_OUTPUT OUT = (Depth_PS_OUTPUT) 0;
	//simply initialize the depth
	OUT.depth = 1.0f;

	return OUT;
}



/********************************************************************************
 *                                                                              *
 *                          Ray Casting vertex shaders                          *
 *                                                                              *
 ********************************************************************************/

//-----------------------------------------------------------------------------
// First hit Vertex shader
//-----------------------------------------------------------------------------
FirstPass_VS_OUTPUT FirstHit_VS(FirstPass_VS_INPUT IN)
{
	FirstPass_VS_OUTPUT OUT = (FirstPass_VS_OUTPUT) 0;

  float4 pos = IN.position;
  float4 ray = pos - matViewInverse[3];
  pos = mul(pos, matView);
  pos = mul(pos, matProjection);

  OUT.position  = pos;
  OUT.texcoord0 = IN.texcoord0;
  OUT.texcoord1 = IN.position;
  OUT.texcoord2 = OUT.position;
  OUT.ray = ray;

  return OUT;
}

//-----------------------------------------------------------------------------
// Ray Casting Vertex shader
//-----------------------------------------------------------------------------
RayCasting_VS_OUTPUT RayCasting_VS(RayCasting_VS_INPUT IN)
{
	RayCasting_VS_OUTPUT OUT = (RayCasting_VS_OUTPUT) 0;

	// transform position and eye to camera space.
	float4 pos = IN.position;
	float4 ray = (pos - eyeLocal);
//	float4 ray = (pos - matViewInverse[3]);

	// transform rays to World space
	ray = mul(ray, matViewInverseTransposedInverse);
//	ray = mul(ray, matProjectionInverseTransposedInverse);

	OUT.position  = IN.position;

	// correct the W perspective division
	OUT.position.z = 0.8 * OUT.position.w;

	OUT.texCoord0 = IN.texCoord0;

	OUT.texCoord1 = ray;
	OUT.tEye      = matViewInverse[3];
//	OUT.diffuse   = IN.diffuse;

	return OUT;
}


#ifdef HLSL_SHADERS

/********************************************************************************
 *																				*
 *						HLSL Ray Casting fragment shaders						*
 *																				*
 ********************************************************************************/


//-----------------------------------------------------------------------------
// First hit calculation pixel shader
//-----------------------------------------------------------------------------
FirstPass_PS_OUTPUT FirstHit_PS(FirstPass_PS_INPUT IN)
{
	//	c16		// pass color - used to store dx, 2dx
	//	OUTPUT: 
	//			.xy - currentCell
	//			.z  - scalar (s(x))
	//			.w  - lambda

	FirstPass_PS_OUTPUT OUT = (FirstPass_PS_OUTPUT) 0;

	float4 result = float4(IN.texcoord0.rg, 0.0, 1.0);

	// Retrieve data from mesh textures
	float4 vertex0, vertex1, vertex2, vertex3;
	float4 normal0, normal1, normal2, normal3;

	vertex0 = tex2D(newVertexSampler, result.xy);
	normal0 = tex2D(newNormalsSampler, result.xy);

	result.x += passColor.x;
	vertex1 = tex2D(newVertexSampler, result.xy);
	normal1 = tex2D(newNormalsSampler, result.xy);

	result.x += passColor.x;
	vertex2 = tex2D(newVertexSampler, result.xy);
	normal2 = tex2D(newNormalsSampler, result.xy);

	vertex3 = float4(vertex0.w, vertex1.w, vertex2.w, 0.0) - matViewInverse[3];
	normal3 = float4(normal0.w, normal1.w, normal2.w, 0.0);

	vertex0 -= matViewInverse[3];
	vertex1 -= matViewInverse[3];
	vertex2 -= matViewInverse[3];

	// Normalize the interpolated ray direction
	float3 ray = normalize(IN.ray.xyz);

	// Compute intersections of the faces of the tetrahedron against the
	// ray, discarding the one that the ray enters the tetrahedron
	float4 num = float4(dot(vertex3.xyz, normal0.xyz), dot(vertex2.xyz, normal1.xyz),
					dot(vertex1.xyz, normal2.xyz), dot(vertex0.xyz, normal3.xyz));
	float4 den = float4(dot(ray.xyz, normal0.xyz), dot(ray.xyz, normal1.xyz),
						dot(ray.xyz, normal2.xyz), dot(ray.xyz, normal3.xyz));

	// test to discard negative denominators
	float4 lambda = num / den;
	lambda = (den < 0 && lambda > 0) ? lambda : 0;

	result.z = (lambda.x > lambda.y) ? lambda.x : lambda.y;
	result.z = (lambda.z > result.z) ? lambda.z : result.z;
	result.z = (lambda.w > result.z) ? lambda.w : result.z;

	//scalar value
	float4 grad = tex2D(newNeighborSampler, result.xy);	//third component of this texture
	float3 x = result.z * ray + matViewInverse[3];
	float gtx = dot(grad.xyz, x);
	result.w = gtx + grad.w;

	result.xy = IN.texcoord0.xy;
	//save the tetrahedral index
	OUT.currentCell = result;

	return OUT;
}


//-----------------------------------------------------------------------------
// Depth peeling pixel shader
//-----------------------------------------------------------------------------
DepthPeeling_PS_OUTPUT DepthPeeling_PS(FirstPass_PS_INPUT IN)
{
	// c16					// pass color - used to store dx, 2dx

	//	OUTPUT: 
	//			.xy - currentCell
	//			.z  - lambda
	//			.w  - scalar (s(x))

	DepthPeeling_PS_OUTPUT OUT = (DepthPeeling_PS_OUTPUT) 0;

	float2 index = (IN.texcoord2.xy/IN.texcoord2.w+1)/2 + 
					float2(0.0001220703125, -0.0001220703125);
	index.y = 1.0 - index.y;
	float4 currentValue = tex2D(currentCellSampler, index);

	float4 result = float4(IN.texcoord0.rg, 0.0, 1.0);

	// Retrieve data from mesh textures
	float4 vertex0, vertex1, vertex2, vertex3;
	float4 normal0, normal1, normal2, normal3;

	vertex0 = tex2D(newVertexSampler, result.xy);
	normal0 = tex2D(newNormalsSampler, result.xy);

	result.x += passColor.x;
	vertex1 = tex2D(newVertexSampler, result.xy);
	normal1 = tex2D(newNormalsSampler, result.xy);

	result.x += passColor.x;
	vertex2 = tex2D(newVertexSampler, result.xy);
	normal2 = tex2D(newNormalsSampler, result.xy);

	vertex3 = float4(vertex0.w, vertex1.w, vertex2.w, 0.0) - matViewInverse[3];
	normal3 = float4(normal0.w, normal1.w, normal2.w, 0.0);

	vertex0 -= matViewInverse[3];
	vertex1 -= matViewInverse[3];
	vertex2 -= matViewInverse[3];

	// Normalize the interpolated ray direction
	float3 ray = normalize(IN.ray.xyz);

	// Compute intersections of the faces of the tetrahedron against the
	// ray, discarding the one that the ray enters the tetrahedron
	float4 num = float4(dot(vertex3.xyz, normal0.xyz), dot(vertex2.xyz, normal1.xyz),
					dot(vertex1.xyz, normal2.xyz), dot(vertex0.xyz, normal3.xyz));
	float4 den = float4(dot(ray.xyz, normal0.xyz), dot(ray.xyz, normal1.xyz),
						dot(ray.xyz, normal2.xyz), dot(ray.xyz, normal3.xyz));

	// test to discard positive denominators
	float4 lambda = num / den;
	lambda = (den < 0 && lambda > 0) ? lambda : 0;

	result.z = (lambda.x > lambda.y) ? lambda.x : lambda.y;
	result.z = (lambda.z > result.z) ? lambda.z : result.z;
	result.z = (lambda.w > result.z) ? lambda.w : result.z;

	// scalar value
	float4 grad = tex2D(newNeighborSampler, result.xy);	//third component of this texture
	float3 x = result.z * ray + matViewInverse[3];
	float gtx = dot(grad.xyz, x);
	result.w = gtx + grad.w;

	result.xy = IN.texcoord0.xy;

	float thisZ = IN.texcoord2.z / IN.texcoord2.w;

	if (result.z > (currentValue.z + 0.01)){
		OUT.currentCell = result;
		OUT.depth = thisZ;
	}else{
		// this value must be written behind all the others
		// set depth value to large value and scalar to -1.0
		OUT.currentCell = float4(0.0, 0.0, 999999.0, 999999.0);
		OUT.depth = 1.0f;
	}

	return OUT;
}


//-----------------------------------------------------------------------------
// Ray Casting Pixel shader 
//-----------------------------------------------------------------------------
RayCasting_PS_OUTPUT RayCasting_PS(RayCasting_PS_INPUT IN)
{
	RayCasting_PS_OUTPUT OUT = (RayCasting_PS_OUTPUT) 0;

	float4 currentValue = tex2D(currentCellSampler, IN.rasterPos.xy);
	float4 result = float4(currentValue.xy, 0.0, 1.0);

	// Retrieve data from mesh textures
	float4 vertex0, vertex1, vertex2, vertex3;
	float4 normal0, normal1, normal2, normal3;
	float4 result1, result2, result3, result4;

	vertex0 = tex2D(newVertexSampler, result.xy);
	normal0 = tex2D(newNormalsSampler, result.xy);
	result1 = tex2D(newNeighborSampler, result.xy);

	result.x += passColor.x;
	vertex1 = tex2D(newVertexSampler, result.xy);
	normal1 = tex2D(newNormalsSampler, result.xy);
	result3 = tex2D(newNeighborSampler, result.xy);

	result.x += passColor.x;
	vertex2 = tex2D(newVertexSampler, result.xy);
	normal2 = tex2D(newNormalsSampler, result.xy);

	vertex3 = float4(vertex0.w, vertex1.w, vertex2.w, 0.0) - matViewInverse[3];
	normal3 = float4(normal0.w, normal1.w, normal2.w, 0.0);

	vertex0 -= matViewInverse[3];
	vertex1 -= matViewInverse[3];
	vertex2 -= matViewInverse[3];

	// Normalize the interpolated ray direction
	float3 ray = normalize(IN.ray.xyz);

	// Compute intersections of the faces of the tetrahedron against the
	// ray, discarding the one that the ray enters the tetrahedron
	float4 num = float4(dot(vertex3.xyz, normal0.xyz), dot(vertex2.xyz, normal1.xyz),
					dot(vertex1.xyz, normal2.xyz), dot(vertex0.xyz, normal3.xyz));
	float4 den = float4(dot(ray.xyz, normal0.xyz), dot(ray.xyz, normal1.xyz),
						dot(ray.xyz, normal2.xyz), dot(ray.xyz, normal3.xyz));

	// changed test to discard negative denominators
	float4 lambda = num / den;
	lambda = (den > 0 && lambda > 0) ? lambda : 999999;

// correct the possible results indexes
	result2.xy = result1.zw;
	result4.xy = result3.zw;

	result1.zw = float2(lambda.x, 0.0);
	result2.zw = float2(lambda.y, 0.0);
	result3.zw = float2(lambda.z, 0.0);
	result4.zw = float2(lambda.w, 0.0);

	// scalar value
	float4 grad = tex2D(newNeighborSampler, result.xy);	//third component of this texture
	vertex2 = tex2D(newVertexSampler, result.xy);

	result = (lambda.x < lambda.y) ? result1 : result2;
	result = (result.z < lambda.z) ? result : result3;
	result = (result.z < lambda.w) ? result : result4;

	float3 x = result.z * ray + matViewInverse[3];

	float gtx = dot(grad.xyz, x);
	result.w = gtx + grad.w;

	// color integration
//	float dif = result.z - currentValue.z;
//	float zCoord = (dif)*lrpParam.z + lrpParam.w;
	float zCoord = result.z - currentValue.z;

	float3 lutIndex = float3(currentValue.w, result.w, zCoord); 

	float4 contribution = tex3D(lutSampler, lutIndex);

	float4 color = tex2D(colorSampler, IN.rasterPos.xy);

	contribution = (currentValue.x < 0.00001) ? float4(0.0, 0.0, 0.0, 0.0) : contribution;
	color += contribution * (1.0 - color.w);

	// if the color contribution exceeds 0.95, eliminate the tetrahedra
		if(currentValue.x > 0.0 && color.w < 0.95){

		OUT.currentCell = result;

	}else{

		OUT.currentCell = float4(0.0, 0.0, 999999.0, -1.0);
	}

	OUT.color = color;

	return OUT;
}


//-----------------------------------------------------------------------------
// Ray Casting Pixel shader 
//-----------------------------------------------------------------------------
RayCasting_PS_OUTPUT RayCastingTF2D_PS(RayCasting_PS_INPUT IN)
{
	RayCasting_PS_OUTPUT OUT = (RayCasting_PS_OUTPUT) 0;
	float4 currentValue = tex2D(currentCellSampler, IN.rasterPos.xy);
	float4 result = float4(currentValue.xy, 0.0, 1.0);

  // Retrieve data from mesh textures
	float4 vertex0, vertex1, vertex2, vertex3;
	float4 normal0, normal1, normal2, normal3;
	float4 result1, result2, result3, result4;

	vertex0 = tex2D(newVertexSampler, result.xy);
	normal0 = tex2D(newNormalsSampler, result.xy);
	result1 = tex2D(newNeighborSampler, result.xy);

	result.x += passColor.x;
	vertex1 = tex2D(newVertexSampler, result.xy);
	normal1 = tex2D(newNormalsSampler, result.xy);
	result3 = tex2D(newNeighborSampler, result.xy);

	result.x += passColor.x;
	vertex2 = tex2D(newVertexSampler, result.xy);
	normal2 = tex2D(newNormalsSampler, result.xy);

	vertex3 = float4(vertex0.w, vertex1.w, vertex2.w, 0.0) - matViewInverse[3];
	normal3 = float4(normal0.w, normal1.w, normal2.w, 0.0);

	vertex0 -= matViewInverse[3];
	vertex1 -= matViewInverse[3];
	vertex2 -= matViewInverse[3];

	// Normalize the interpolated ray direction
	float3 ray = normalize(IN.ray.xyz);

	// Compute intersections of the faces of the tetrahedron against the
	// ray, discarding the one that the ray enters the tetrahedron
	float4 num = float4(dot(vertex3.xyz, normal0.xyz), dot(vertex2.xyz, normal1.xyz),
					dot(vertex1.xyz, normal2.xyz), dot(vertex0.xyz, normal3.xyz));
	float4 den = float4(dot(ray.xyz, normal0.xyz), dot(ray.xyz, normal1.xyz),
						dot(ray.xyz, normal2.xyz), dot(ray.xyz, normal3.xyz));

	// test to discard negative denominators
	float4 lambda = num / den;
	lambda = (den > 0 && lambda > 0) ? lambda : 999999;

// correct the possible results indexes
	result2.xy = result1.zw;
	result4.xy = result3.zw;

	result1.zw = float2(lambda.x, 0.0);
	result2.zw = float2(lambda.y, 0.0);
	result3.zw = float2(lambda.z, 0.0);
	result4.zw = float2(lambda.w, 0.0);

	// scalar value
	float4 grad = tex2D(newNeighborSampler, result.xy);	//third component of this texture
	vertex2 = tex2D(newVertexSampler, result.xy);

	result = (lambda.x < lambda.y) ? result1 : result2;
	result = (result.z < lambda.z) ? result : result3;
	result = (result.z < lambda.w) ? result : result4;

	float3 x = result.z * ray + matViewInverse[3];

	float gtx = dot(grad.xyz, x);
	result.w = gtx + grad.w;

	// color integration
	float dif = result.z - currentValue.z;
	float zCoord = (dif)*lrpParam.z + lrpParam.w;

	float3 lutIndex = float3(currentValue.w, result.w, zCoord); 

	float4 contribution = tex2D(lut2DSampler, lutIndex) * dif;

	float4 color = tex2D(colorSampler, IN.rasterPos.xy);

	contribution = (currentValue.x < 0.00001) ? float4(0.0, 0.0, 0.0, 0.0) : contribution;
	color += contribution * (1.0 - color.w);

	// if the color contribution exceeds 0.95, eliminate the tetrahedra
	if(currentValue.x > 0.0 && color.w < 0.95){
		OUT.currentCell = result;

	}else{

		OUT.currentCell = float4(0.0, 0.0, 999999.0, -1.0);
	}

	OUT.color = color;

	return OUT;
}



#else


/********************************************************************************
 *																				*
 *			Assembler version of HLSL vertex and pixel shaders					*
 *		Generated by Microsoft (R) D3DX9 Shader Compiler 4.09.00.1126			*
 *																				*
 ********************************************************************************/

PixelShader RayCasting_PS = asm 
{
//   fxc /T ps_2_0 /E RayCasting_PS RayCasting.fx
//
//
// Parameters:
//
//   sampler2D colorSampler;
//   sampler2D currentCellSampler;
//   float4 matViewInverse[3];
//   float4 lrpParam;
//   sampler3D lutSampler;
//   sampler2D newNeighborSampler;
//   sampler2D newNormalsSampler;
//   sampler2D newVertexSampler;
//   float4 passColor;
//
//
// Registers:
//
//   Name               Reg   Size
//   ------------------ ----- ----
//   lrpParam           c26      1
//   matViewInverse[3]  c27      1
//   passColor          c28      1
//   newVertexSampler   s0       1
//   newNormalsSampler  s1       1
//   newNeighborSampler s2       1
//   currentCellSampler s3       1
//   colorSampler       s4       1
//   lutSampler         s5       1
//

    ps_2_0
    def c0, 0, 1, 999999, -0.00001
    def c1, 0, 0, 999999, -1
    dcl t0.xy
    dcl t1.xyz
    dcl_2d s0
    dcl_2d s1
    dcl_2d s2
    dcl_2d s3
    dcl_2d s4
    dcl_volume s5
    texld r2, t0, s3
    add r0.y, r2.x, c28.x
    mov r0.z, r2.y
    mov r11.xy, r0.yzxw
    add r0.x, r0.y, c28.x
    mov r0.y, r0.z
    texld r5, r11, s0
    texld r3, r0, s0
    texld r1, r2, s0
    texld r4, r2, s1
    texld r6, r0, s1
    texld r0, r0, s2
    add r9.y, r5.w, -c27.y
    add r8.xyz, r5, -c27
    add r9.z, r3.w, -c27.z
    add r7.xyz, r3, -c27
    add r9.x, r1.w, -c27.x
    add r10.xyz, r1, -c27
    dp3 r3.x, r9, r4
    mov r1.z, r6.w
    texld r5, r11, s1
    mov r1.y, r5.w
    mov r1.x, r4.w
    nrm r9.xyz, t1
    dp3 r1.w, r9, r1
    dp3 r3.w, r10, r1
    dp3 r1.z, r9, r6
    dp3 r3.z, r8, r6
    dp3 r1.y, r9, r5
    dp3 r3.y, r7, r5
    dp3 r1.x, r9, r4
    rcp r4.x, r1.x
    rcp r4.y, r1.y
    rcp r4.z, r1.z
    rcp r4.w, r1.w
    cmp r1, -r1, c0.x, c0.y
    mul r3, r3, r4
    cmp r4, -r3, c0.x, c0.y
    mul r1, r1, r4
    cmp r5, -r1, c0.z, r3
    min r8.z, r5.y, r5.x
    min r7.z, r5.z, r8.z
    min r3.z, r5.w, r7.z
    mad r1.xyz, r3.z, r9, c27
    dp3 r0.x, r0, r1
    add r3.w, r0.w, r0.x
    mov r0.y, r3.w
    add r0.w, -r2.z, r3.z
    mad r0.z, r0.w, c26.z, c26.w
    mov r0.x, r2.w
    texld r4, r11, s2
    texld r6, r2, s2
    texld r0, r0, s5
    texld r1, t0, s4
    add r2.w, -r5.y, r5.x
    cmp r8.x, r2.w, r6.z, r6.x
    cmp r8.y, r2.w, r6.w, r6.y
    add r2.w, -r5.z, r8.z
    cmp r7.xy, r2.w, r4, r8
    add r2.w, -r5.w, r7.z
    cmp r3.x, r2.w, r4.z, r7.x
    cmp r3.y, r2.w, r4.w, r7.y
    cmp r3, -r2.x, c1, r3
    mov oC0, r3
    add r2.w, r2.x, c0.w
    cmp r0, r2.w, r0, c0.x
    add r2.w, -r1.w, c0.y
    mad r0, r0, r2.w, r1
    mov oC1, r0

// approximately 71 instruction slots used (12 texture, 59 arithmetic)
};



PixelShader RayCastingTF2D_PS = asm
{
// Parameters:
//
//   sampler2D colorSampler;
//   sampler2D currentCellSampler;
//   float4 matViewInverse[3];
//   sampler2D lut2DSampler;
//   sampler2D newNeighborSampler;
//   sampler2D newNormalsSampler;
//   sampler2D newVertexSampler;
//   float4 passColor;
//
//
// Registers:
//
//   Name               Reg   Size
//   ------------------ ----- ----
//   matViewInverse[3]  c27      1
//   passColor          c28      1
//   newVertexSampler   s0       1
//   newNormalsSampler  s1       1
//   newNeighborSampler s2       1
//   currentCellSampler s3       1
//   colorSampler       s4       1
//   lut2DSampler       s5       1
//

    ps_2_0
    def c0, 0, 1, 999999, -0.00001
    def c1, -0.95, 0, 0, 0
    def c2, 0, 0, 999999, -1
    dcl t0.xy
    dcl t1.xyz
    dcl_2d s0
    dcl_2d s1
    dcl_2d s2
    dcl_2d s3
    dcl_2d s4
    dcl_2d s5
    texld r2, t0, s3
    add r0.y, r2.x, c28.x
    mov r0.z, r2.y
    mov r11.xy, r0.yzxw
    add r1.x, r0.y, c28.x
    mov r1.y, r0.z
    texld r5, r11, s0
    texld r3, r1, s0
    texld r0, r2, s0
    texld r4, r2, s1
    texld r6, r1, s1
    texld r1, r1, s2
    add r9.y, r5.w, -c27.y
    add r8.xyz, r5, -c27
    add r9.z, r3.w, -c27.z
    add r7.xyz, r3, -c27
    add r9.x, r0.w, -c27.x
    add r10.xyz, r0, -c27
    dp3 r3.x, r9, r4
    mov r0.z, r6.w
    texld r5, r11, s1
    mov r0.y, r5.w
    mov r0.x, r4.w
    nrm r9.xyz, t1
    dp3 r0.w, r9, r0
    dp3 r3.w, r10, r0
    dp3 r0.z, r9, r6
    dp3 r3.z, r8, r6
    dp3 r0.y, r9, r5
    dp3 r3.y, r7, r5
    dp3 r0.x, r9, r4
    rcp r4.x, r0.x
    rcp r4.y, r0.y
    rcp r4.z, r0.z
    rcp r4.w, r0.w
    cmp r0, -r0, c0.x, c0.y
    mul r3, r3, r4
    cmp r4, -r3, c0.x, c0.y
    mul r0, r0, r4
    cmp r6, -r0, c0.z, r3
    min r8.z, r6.y, r6.x
    min r7.z, r6.z, r8.z
    min r0.z, r6.w, r7.z
    mad r3.xyz, r0.z, r9, c27
    dp3 r0.x, r1, r3
    add r0.w, r1.w, r0.x
    mov r0.y, r0.w
    mov r0.x, r2.w
    texld r1, r11, s2
    texld r4, r0, s5
    texld r5, t0, s4
    texld r3, r2, s2
    add r7.w, -r6.z, r8.z
    add r6.w, -r6.w, r7.z
    add r2.w, -r6.y, r6.x
    add r8.w, -r2.z, r0.z
    mul r4, r4, r8.w
    add r8.w, r2.x, c0.w
    cmp r4, r8.w, r4, c0.x
    add r8.w, -r5.w, c0.y
    mad r4, r4, r8.w, r5
    add r5.w, r4.w, c1.x
    mov oC1, r4
    cmp r4.w, r5.w, c0.x, c0.y
    cmp r8.x, r2.w, r3.z, r3.x
    cmp r8.y, r2.w, r3.w, r3.y
    cmp r2.w, -r2.x, c0.x, c0.y
    cmp r7.xy, r7.w, r1, r8
    mul r2.w, r4.w, r2.w
    cmp r0.x, r6.w, r1.z, r7.x
    cmp r0.y, r6.w, r1.w, r7.y
    cmp r0, -r2.w, c2, r0
    mov oC0, r0

// approximately 75 instruction slots used (12 texture, 63 arithmetic)
};


PixelShader FirstHit_PS = asm
{
// Parameters:
//
//   float4 matViewInverse[3];
//   sampler2D newNeighborSampler;
//   sampler2D newNormalsSampler;
//   sampler2D newVertexSampler;
//   float4 passColor;
//
//
// Registers:
//
//   Name               Reg   Size
//   ------------------ ----- ----
//   matViewInverse[3]  c27      1
//   passColor          c28      1
//   newVertexSampler   s0       1
//   newNormalsSampler  s1       1
//   newNeighborSampler s2       1
//

    ps_2_0
    def c0, 0, 1, 0, 0
    dcl t0.xy
    dcl t3.xyz
    dcl_2d s0
    dcl_2d s1
    dcl_2d s2
    texld r0, t0, s0
    add r6.x, r0.w, -c27.x
    add r9.xyz, r0, -c27
    mov r0.y, t0.y
    add r0.x, t0.x, c28.x
    add r1.x, r0.x, c28.x
    mov r1.y, t0.y
    texld r2, r0, s0
    texld r4, r0, s1
    texld r0, r1, s0
    texld r3, t0, s1
    texld r5, r1, s1
    texld r1, r1, s2
    add r6.y, r2.w, -c27.y
    add r8.xyz, r2, -c27
    add r6.z, r0.w, -c27.z
    add r7.xyz, r0, -c27
    dp3 r2.x, r6, r3
    mov r0.y, r4.w
    mov r0.x, r3.w
    mov r0.z, r5.w
    nrm r6.xyz, t3
    dp3 r0.w, r6, r0
    dp3 r2.w, r9, r0
    dp3 r0.z, r6, r5
    dp3 r2.z, r8, r5
    dp3 r0.y, r6, r4
    dp3 r2.y, r7, r4
    dp3 r0.x, r6, r3
    rcp r3.x, r0.x
    rcp r3.y, r0.y
    rcp r3.z, r0.z
    rcp r3.w, r0.w
    cmp r0, r0, c0.x, c0.y
    mul r2, r2, r3
    cmp r3, -r2, c0.x, c0.y
    mul r0, r0, r3
    cmp r2, -r0, c0.x, r2
    max r0.w, r2.y, r2.x
    max r3.w, r0.w, r2.z
    max r0.z, r3.w, r2.w
    mad r2.xyz, r0.z, r6, c27
    dp3 r0.x, r1, r2
    add r0.w, r1.w, r0.x
    mov r0.xy, t0
    mov oC0, r0

// approximately 48 instruction slots used (7 texture, 41 arithmetic)
};


PixelShader DepthPeeling_PS = asm
{
// Parameters:
//
//   sampler2D currentCellSampler;
//   float4 matViewInverse[3];
//   sampler2D newNeighborSampler;
//   sampler2D newNormalsSampler;
//   sampler2D newVertexSampler;
//   float4 passColor;
//
//
// Registers:
//
//   Name               Reg   Size
//   ------------------ ----- ----
//   matViewInverse[3]              c27      1
//   passColor          c28      1
//   newVertexSampler   s0       1
//   newNormalsSampler  s1       1
//   newNeighborSampler s2       1
//   currentCellSampler s3       1
//

    ps_2_0
    def c0, 1, 0, 0.01, 0
    def c1, 0.000976563, -0.000976563, 0.5, 0
    def c2, 0, 0, 999999, 999999
    dcl t0.xy
    dcl t2
    dcl t3.xyz
    dcl_2d s0
    dcl_2d s1
    dcl_2d s2
    dcl_2d s3
    texld r0, t0, s0
    add r7.x, r0.w, -c27.x
    add r10.xyz, r0, -c27
    mov r0.y, t0.y
    add r0.x, t0.x, c28.x
    add r2.x, r0.x, c28.x
    mov r2.y, t0.y
    rcp r7.w, t2.w
    mad r1.xy, t2, r7.w, c0.x
    mad r1.xy, r1, c1.z, c1
    add r1.y, -r1.y, c0.x
    texld r3, r0, s0
    texld r5, r0, s1
    texld r0, r2, s0
    texld r4, t0, s1
    texld r6, r2, s1
    texld r2, r2, s2
    texld r1, r1, s3
    add r7.y, r3.w, -c27.y
    add r9.xyz, r3, -c27
    add r7.z, r0.w, -c27.z
    add r8.xyz, r0, -c27
    dp3 r3.x, r7, r4
    mov r0.y, r5.w
    mov r0.x, r4.w
    mov r0.z, r6.w
    nrm r7.xyz, t3
    dp3 r0.w, r7, r0
    dp3 r3.w, r10, r0
    dp3 r0.z, r7, r6
    dp3 r3.z, r9, r6
    dp3 r0.y, r7, r5
    dp3 r3.y, r8, r5
    dp3 r0.x, r7, r4
    rcp r4.x, r0.x
    rcp r4.y, r0.y
    rcp r4.z, r0.z
    rcp r4.w, r0.w
    cmp r0, r0, c0.y, c0.x
    mul r3, r3, r4
    cmp r4, -r3, c0.y, c0.x
    mul r0, r0, r4
    cmp r3, -r0, c0.y, r3
    max r0.w, r3.y, r3.x
    max r1.w, r0.w, r3.z
    max r0.z, r1.w, r3.w
    mad r3.xyz, r0.z, r7, c27
    dp3 r0.x, r2, r3
    add r0.w, r2.w, r0.x
    mul r2.w, r7.w, t2.z
    add r1.w, r1.z, c0.z
    add r1.w, -r0.z, r1.w
    mov r0.xy, t0
    cmp r0, r1.w, c2, r0
    cmp r1.w, r1.w, c0.x, r2.w
    mov oC0, r0
    mov oDepth, r1.w

// approximately 59 instruction slots used (8 texture, 51 arithmetic)
};

#endif


//*****************************************************************************
// TECHNIQUES
//*****************************************************************************

//-----------------------------------------------------------------------------
// technique RenderDepthPeeling
//-----------------------------------------------------------------------------
technique RenderDepthPeeling {

  pass p0 {

    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    VertexShader = compile vs_2_0 FaceColored_VS();
    PixelShader  = compile ps_2_0 RenderDepthPeeling_PS();
  }
}


//-----------------------------------------------------------------------------
// technique OcclusionTest
//-----------------------------------------------------------------------------
technique OcclusionTest {

	pass p0 {

		Zenable = true;
		ZWriteEnable = true;
		CullMode = None;
		ShadeMode = Flat;
		FillMode = Solid;
		AlphaBlendEnable = False;
    AlphaTestEnable = False;

		VertexShader = compile vs_2_0 RayCasting_VS();
		PixelShader  = compile ps_2_0 Depth_PS();
    }

	pass p1 {

		Zenable = true;
		CullMode = None;
		ShadeMode = Flat;
    FillMode = Solid;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;

		VertexShader = compile vs_2_0 RayCasting_VS();
		PixelShader  = compile ps_2_0 PassThrough_PS();
	}
}


//-----------------------------------------------------------------------------
// technique FaceColoredRender
//-----------------------------------------------------------------------------
technique FaceColoredRender {

  pass p0 {

    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    VertexShader = compile vs_2_0 FaceColored_VS();
    PixelShader  = compile ps_2_0 FaceColored_PS();
  }
}


//-----------------------------------------------------------------------------
// technique ShowColorBuffer
//-----------------------------------------------------------------------------
technique ShowColorBuffer {

  pass p0 {

    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = True;
    VertexShader = compile vs_2_0 ShowColorBuffer_VS();
    PixelShader  = compile ps_2_0 ShowColorBuffer_PS();
  }
}


#ifdef HLSL_SHADERS

/********************************************************************************
 *																				*
 *				Techniques for HLSL vertex and pixel shaders					*
 *																				*
 ********************************************************************************/

//-----------------------------------------------------------------------------
// technique FirstHit
//-----------------------------------------------------------------------------
technique FirstHit {

  pass p0 {   

    Zenable = true;
    ZFunc = LESSEQUAL;
    CullMode = CW;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    VertexShader = compile VS_COMPILER_TARGET FirstHit_VS();
		PixelShader  = compile PS_COMPILER_TARGET FirstHit_PS();
  }
}

//-----------------------------------------------------------------------------
// technique DepthPeeling
//-----------------------------------------------------------------------------
technique DepthPeeling {

  pass p0 {   
  
		Zenable = true;
		ZWriteEnable = true;
		ZFunc = LESSEQUAL;
    CullMode = CW;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;

    VertexShader = compile VS_COMPILER_TARGET FirstHit_VS();
    PixelShader  = compile PS_COMPILER_TARGET DepthPeeling_PS();
  }
}

//-----------------------------------------------------------------------------
// technique RayCasting
//-----------------------------------------------------------------------------
technique RayCasting {

  pass p0 {
  
		Zenable = true;
		ZWriteEnable = false;
    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;
		VertexShader = compile VS_COMPILER_TARGET RayCasting_VS();
    PixelShader  = compile PS_COMPILER_TARGET RayCasting_PS();
  }
}

//-----------------------------------------------------------------------------
// technique RayCasting
//-----------------------------------------------------------------------------
technique RayCastingTF2D {

  pass p0 {
  
		Zenable = true;
		ZWriteEnable = false;
    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;
    VertexShader = compile VS_COMPILER_TARGET RayCasting_VS();
    PixelShader  = compile PS_COMPILER_TARGET RayCastingTF2D_PS();
  }
}


#else

/********************************************************************************
 *																				*
 *		Techniques for assembler version of HLSL vertex and pixel shaders		*
 *																				*
 ********************************************************************************/


//-----------------------------------------------------------------------------
// technique RayCasting
//-----------------------------------------------------------------------------
technique RayCasting {

    pass p0 {

		Zenable = true;
		ZWriteEnable = false;
    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;

    PixelShaderConstant[26] = <lrpParam>;
		PixelShaderConstant[27] = <matViewInverse[3]>;

		PixelShaderConstant[28] = <passColor>;

		Texture[0] = <tNewVertex>;
		Texture[1] = <tNewNormals>;
		Texture[2] = <tNewNeighbor>;
		Texture[3] = <tCurrentCell>;
		Texture[4] = <tColor>;
		Texture[5] = <tLut>;

		VertexShader = compile vs_2_0 RayCasting_VS();
		PixelShader  =  (RayCasting_PS);
  }
}


//-----------------------------------------------------------------------------
// technique RayCastingTF2D
//-----------------------------------------------------------------------------
technique RayCastingTF2D {

  pass p0 {
		Zenable = true;
		ZWriteEnable = false;
    CullMode = None;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;

		PixelShaderConstant[27] = <matViewInverse[3]>;

		PixelShaderConstant[28] = <passColor>;

		Texture[0] = <tNewVertex>;
		Texture[1] = <tNewNormals>;
		Texture[2] = <tNewNeighbor>;
		Texture[3] = <tCurrentCell>;
		Texture[4] = <tColor>;
		Texture[5] = <tLut2D>;

		VertexShader = compile vs_2_0 RayCasting_VS();
		PixelShader  =  (RayCastingTF2D_PS);
  }
}


//-----------------------------------------------------------------------------
// technique FirstHit
//-----------------------------------------------------------------------------
technique FirstHit {

  pass p0 {

		Zenable = true;
		ZFunc = LESSEQUAL;
		CullMode = CW;
		FillMode = Solid;
		ShadeMode = Flat;
		AlphaBlendEnable = False;

		PixelShaderConstant[27] = <matViewInverse[3]>;

		PixelShaderConstant[28] = <passColor>;

		Texture[0] = <tNewVertex>;
		Texture[1] = <tNewNormals>;
		Texture[2] = <tNewNeighbor>;

    VertexShader = compile vs_2_0 FirstHit_VS();
		PixelShader  = (FirstHit_PS);
  }
}


//-----------------------------------------------------------------------------
// technique DepthPeeling
//-----------------------------------------------------------------------------
technique DepthPeeling {

  pass p0 {

	  Zenable = true;
	  ZWriteEnable = true;
	  ZFunc = LESSEQUAL;
    CullMode = CW;
    FillMode = Solid;
    ShadeMode = Flat;
    AlphaBlendEnable = False;
    AlphaTestEnable = False;

    PixelShaderConstant[27] = <matViewInverse[3]>;

	  PixelShaderConstant[28] = <passColor>;

	  Texture[0] = <tNewVertex>;
	  Texture[1] = <tNewNormals>;
	  Texture[2] = <tNewNeighbor>;
	  Texture[3] = <tCurrentCell>;

    VertexShader = compile vs_2_0 FirstHit_VS();
    PixelShader  = (DepthPeeling_PS);
  }
}


#endif