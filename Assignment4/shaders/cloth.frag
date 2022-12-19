#version 450

layout(binding = 8) uniform sampler2D gTex;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

const vec3 lightPos = vec3(0.5, 0.5, 0.7);
const vec3 lightIntensity = vec3(0.15);

void main() {
	outColor = vec4(texture(gTex, texCoord));
}