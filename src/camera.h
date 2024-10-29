
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
	glm::vec3 _direction;

	float fov = 60.0f;
	float aspect = 1.0f;
	float near = 0.01f;
	float far = 2000.0f;

	float move_speed = 1.0;

	constexpr static glm::vec3 UP{0, 1.0, 0};
	constexpr static glm::vec3 DOWN{0, -1.0, 0};
	constexpr static glm::vec3 LEFT{-1.0, 0, 0};
	constexpr static glm::vec3 RIGHT{1.0, 0, 0};

	Camera() = default;
	Camera(const glm::vec3& pos, const glm::vec3& up, const glm::vec3& center);
	GPUCameraData GetGPUData();

	glm::mat4 get_view();
	glm::mat4 get_perspection();

	void move_position(glm::vec3 direction);
};
