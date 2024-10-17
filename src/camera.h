
#include "glm/ext/vector_float3.hpp"

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 projview;
};
class Camera {
public:
	glm::vec3 _position;
	glm::vec3 _up;
	glm::vec3 _center;

	float fov = 60.0f;
	float aspect = 1.0f;
	float near = 10.0f;
	float far = 100.0f;

	Camera() = default;
	Camera(const glm::vec3& pos, const glm::vec3& up, const glm::vec3& center);
	GPUCameraData GetGPUData();

	glm::mat4 get_view();
	glm::mat4 get_perspection();
};
