#pragma once
#include"config.hpp"

//相机的外参信息和旋转信息
class ViewInfo
{
public:
	float center[3];
	float rotation[9];//这个矩阵是内参矩阵左乘旋转矩阵得到的
	cudaTextureObject_t texObj;//cuda形式的纹理
	cudaArray_t cuArray;//这个东西和texObj需要共同存在，但实际读取像素的时候并不会用到它

	//和当前的相机共同构成相机组的其它相机
	uint16_t imgGroup[IMG_GROUP_SIZE];
};