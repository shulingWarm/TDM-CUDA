#pragma once
#include"scene.hpp"
#include<vector>
#include"cudaHandleError.hpp"

//每个线程块的大小
#define BLOCK_THREAD 64

//从scene里面获取z值的数组
__global__ void copyHeight(Scene* cudaScene,float* cudaHeight)
{
	//当前线程对应的x值
	int xLocal=blockIdx.x*BLOCK_THREAD + threadIdx.x;
	//判断x的位置是否超过界限，为了加快复制的节奏才这样操作的
	if(xLocal < cudaScene->grid->gridInfo.gridSize[0])
	{
		//复制对应位置的z值
		cudaHeight[blockIdx.y * cudaScene->grid->gridInfo.gridSize[0] + xLocal]=
			cudaScene->grid->getDataAt(xLocal,blockIdx.y)->z_;
	}
}

//输入一个cuda版本的scene,把它保存在cpu的版本里面
void exportSceneHeight(Scene* cudaScene,Scene* cpuScene,std::vector<float>& heightMap)
{
	//网格单元的大小相关的信息
	auto& gridInfo=cpuScene->grid->gridInfo;
	//开辟对应大小的cuda空间，为了方便传播就直接做成数组了
	float* cudaHeight;
	handleError(cudaMalloc((void**)&cudaHeight,sizeof(float)*gridInfo.gridSize[0]*gridInfo.gridSize[1]));
	copyHeight<<<dim3(gridInfo.gridSize[0]/BLOCK_THREAD+1,gridInfo.gridSize[1],1),BLOCK_THREAD>>>(cudaScene,cudaHeight);
	//把cuda的高度图复制到cpu端
	heightMap.resize(gridInfo.gridSize[0]*gridInfo.gridSize[1]);
	handleError(cudaMemcpy(heightMap.data(),cudaHeight,sizeof(float)*heightMap.size(),cudaMemcpyDeviceToHost));
	//释放cuda内存
	handleError(cudaFree(cudaHeight));
}