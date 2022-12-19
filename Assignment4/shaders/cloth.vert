#version 450

layout(push_constant) uniform PushStruct {
	mat4 mvp;
	vec4 viewPos;
	float dT; // timestep	
} p;

layout(binding = 0) readonly buffer PosArray {vec4 gPosArray[];};
layout(binding = 4) readonly buffer Tex {vec2 gTex[];};
layout(binding = 6) readonly buffer Indices {uint gIndices[];};

layout(location = 0) out vec3 pos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 texCoord;

void main() {
    uint idx = gIndices[gl_VertexIndex];
    pos = gPosArray[idx].xyz;
    texCoord = gTex[idx];
	gl_Position = p.mvp * vec4(pos, 1);
}