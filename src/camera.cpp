#include "camera.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"


Camera::Camera(const glm::vec3& pos, const glm::vec3& up, const glm::vec3& direction)
	:_position(pos), _up(up), _direction(direction)
{

}

glm::mat4 Camera::get_view()
{
	return glm::lookAtLH(_position, _position + _direction, _up);
}

glm::mat4 Camera::get_perspection()
{
	glm::mat4 project = glm::perspectiveLH_ZO(glm::radians(fov), aspect, near, far);
	project[1][1] *= -1.0f;

	return project;
}

GPUCameraData Camera::GetGPUData()
{
	GPUCameraData data = {
		.view = get_view(),
		.proj = get_perspection(),
	};
	data.projview = data.proj * data.view;

	return data;
}

void Camera::move_position(glm::vec3 direction)
{
	_position += move_speed * direction;
}
