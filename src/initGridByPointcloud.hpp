#pragma once
#include<vector>
#include"scene.hpp"
#include"cudaHandleError.hpp"
#include"config.hpp"
#include"userAtomic.hpp"

//其实就是遍历每个点，然后把点放进栅格单元里面
__global__ void kernelPointcloudInit(float* pointcloud,Scene* cudaScene,unsigned pointNum)
{
	//计算偏移量
	unsigned blockOffset=blockDim.x*blockIdx.x + threadIdx.x;
	if(blockOffset>=pointNum) return;
	//如果超过了总的点数就不需要处理了
	//当前位置的点
	float* blockPoint=pointcloud+(blockOffset*3);
	//把当前的点坐标变换到离散坐标
	int xy[2];
	xy[0]=cudaScene->grid->gridInfo.toGridCoord(0,blockPoint[0]);
	xy[1]=cudaScene->grid->gridInfo.toGridCoord(1,blockPoint[1]);
	//判断这个数据是否在网格的范围内
	if(cudaScene->grid->gridInfo.isPointInRange(xy))
	{
		//更新目标位置的z值
		floatAtomicMax(&(cudaScene->grid->getDataAt(xy[0],xy[1])->z_),blockPoint[2]);
	}
}

//把有高度的地方标记成可用的分数
__global__ void labelSeedCells(Scene* cudaScene)
{
	//获取当前位置负责的网格单元
	DOMCell* blockCell=cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y);
	//如果目前单元的高度是一个合法的高度，就把它的分数弄成可用的分数
	if(blockCell->z_>0)
	{
		blockCell->score=PRIOR_SCORE;
		//每个网格单元的可使用次数明明是初始化过的，但复制的时候没有生效
		blockCell->refTime=INIT_REF;
	}
	else
	{
		blockCell->score=-100;
		blockCell->refTime=0;
	}
	//这里穿插一下，初始化一下网格单元里面的随机数种子
	curand_init(0, blockIdx.y*gridDim.x+blockIdx.x, 0, 
			&(cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->randState));
}

//用点云初始化scene,这个时候传进来的就已经是cuda scene了
void initSceneByPointcloud(std::vector<float>& pointcloud,Scene* cudaScene,
	Scene* cpuScene)
{
	//把点云数据转换成cuda
	float* cudaPointcloud;
	handleError(cudaMalloc((void**)&cudaPointcloud,sizeof(float)*pointcloud.size()));
	handleError(cudaMemcpy(cudaPointcloud,pointcloud.data(),sizeof(float)*pointcloud.size(),cudaMemcpyHostToDevice));
	//点云的数量
	unsigned pointNum=pointcloud.size()/3;
	//进入核函数，把每个点投影到目标区域里面
	kernelPointcloudInit<<<pointNum/64+1,64>>>(cudaPointcloud,cudaScene,pointNum);
	cudaDeviceSynchronize();
	//网格的大小
	auto* gridInfo=&cpuScene->grid->gridInfo;
	//对于具有正常高度的区域，把它的分数标记成初始分数
	labelSeedCells<<<dim3(gridInfo->gridSize[0],gridInfo->gridSize[1],1),1>>>(cudaScene);
	//处理完之后就可以把点云数据释放了
	handleError(cudaFree(cudaPointcloud));
}