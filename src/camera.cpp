#include "camera.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"


Camera::Camera(const glm::vec3& pos, const glm::vec3& up, const glm::vec3& direction)
	:_position(pos), _up(up), _center(direction)
{

}

glm::mat4 Camera::get_view()
{
	return glm::lookAt(_position, _center, _up);
}

glm::mat4 Camera::get_perspection()
{
	return glm::perspective(glm::radians(fov), aspect, near, far);
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
