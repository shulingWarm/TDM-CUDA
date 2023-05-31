#pragma once

//处理cuda的报错信息
void handleError(cudaError_t statu)
{
	if(statu!=cudaSuccess)
	{
	  std::cerr<<cudaGetErrorString(statu)<<std::endl;
	}
}