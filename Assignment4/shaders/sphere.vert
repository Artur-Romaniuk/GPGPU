#version 450

layout(push_constant) uniform PushStruct {
  	mat4 mvp;
  	uvec4 nParts_w_h; // [0]=nParticles, [1]=width, [2]=height
	vec4 dTs_rest; // timestep, old timestep, total, restDistance
	vec4 ballPos;
} p;

layout(binding = 9) readonly buffer TriangleSoup {vec4 gTriangleSoup[];};
layout(binding = 10) readonly buffer NormalSoup {vec4 gNormals[];};

layout(location = 0) out vec3 pos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 albedo;

void main() {
	pos = gTriangleSoup[gl_VertexIndex].xyz * p.ballPos.w + p.ballPos.xyz;
	normal = gNormals[gl_VertexIndex].xyz;
	albedo = vec3(0.6, 0.2, 0.2);
	gl_Position = p.mvp * vec4(pos, 1);

}