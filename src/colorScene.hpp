#pragma once
#include"scene.hpp"
#include"geometry.hpp"
#define FLOAT_INT_RATE 1000
#define SQRT2 1.414213562

//根据方向顺序初始化xy的偏移方向
__device__ void getXyOffsetByDirection(int* xyOffset,int idDirection)
{
	//x增加的方向
	if(idDirection==0 || idDirection==1 || idDirection==7)
	{
		xyOffset[0]=1;
	}
	else if(idDirection == 2 || idDirection == 6)
	{
		xyOffset[0]=0;
	}
	else
	{
		xyOffset[0]=-1;
	}

	//y增加的方向
	if(idDirection>0 && idDirection<4)
	{
		xyOffset[1]=1;
	}
	else if(idDirection > 4)
	{
		xyOffset[1]=-1;
	}
	else
	{
		xyOffset[1]=0;
	}
}

//线程线程标号得到的偏移量的倍率
__device__ int getOffsetRate(int idStep)
{
	//1 + i*1.5
	//1 2 4 5 7 13.5 
	if(idStep<5)
	{
		return 1.f+(float)idStep*1.5;
	}
	return 1.f + (float)idStep*2.5;
}

//处理每个图片的遮挡 直接全部走一遍检测
__global__ void colorCells(Scene* cudaScene)
{
	//共享内存，准备8个方向的当前网格单元的变化方向
	//它对应的是显存的最小值
	//这其实应该是浮点型的，但为了方便使用原子操作，还是使用uint型了
	__shared__ unsigned directionSlope[IMG_GROUP_SIZE];
	//每个维度的变换比例 虽然这里用的是宏，但宏发生变化之后这里还需要改
	__shared__ uchar offsetRate[IMG_GROUP_SIZE];
	//由每个线程的第1个点把运行方向初始化成0,方便后面比较最大值
	if(threadIdx.y==0)
	{
		directionSlope[threadIdx.x]=0;
	}
	//另外每个步长下的线程负责更新一下自己的倍率
	if(threadIdx.x==0)
	{
		offsetRate[threadIdx.y]=getOffsetRate(threadIdx.y);
	}
	__syncthreads();
	//xy的偏移方向 写一个3是为了后面复用这块内存，后面会有一个三维点用这个地方
	int xyOffset[3];
	getXyOffsetByDirection(xyOffset,threadIdx.x);
	//根据信息计算实际的xy的位置
	xyOffset[0]=blockIdx.x + xyOffset[0]*offsetRate[threadIdx.y];
	xyOffset[1]=blockIdx.y + xyOffset[1]*offsetRate[threadIdx.y];
	//判断一下当前的位置是否在网格的范围内，如果在范围内再判断
	if(cudaScene->grid->gridInfo.isPointInRange(xyOffset))
	{
		//计算临时的斜率
		float tempSlope=(cudaScene->grid->getDataAt(xyOffset[0],xyOffset[1])->z_-
			cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->z_)/offsetRate[threadIdx.y];
		//如果距离是根号2的情况，再把距离除根号2
		if(xyOffset[0]!=0 && xyOffset[1]!=0)
		{
			tempSlope*=SQRT2;
		}
		//为了方便使用原子操作，把tempSlope弄成整型
		tempSlope*=FLOAT_INT_RATE;
		atomicMax(directionSlope+threadIdx.x,(unsigned)tempSlope);
	}
	__syncthreads();
	//判断主方向的相机属于哪个分区
	if(threadIdx.x==0 && threadIdx.y==0)
	{
		//当前的主相机组
		unsigned idGroup=cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->idGroup;
		//当前网格单元对应的三维点，复用之前的内存，直接在这上面写
		((float*)xyOffset)[0]=cudaScene->grid->gridInfo.toWorldCoord(0,blockIdx.x);
		((float*)xyOffset)[1]=cudaScene->grid->gridInfo.toWorldCoord(1,blockIdx.y);
		//记录当前三维点的z值
		((float*)xyOffset)[2]=cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->z_;
		//计算它属于哪个分区
		unsigned mainSegment = getAngleSeg((float*)xyOffset);
		//计算主相机和当前位置的夹角
		float mainSlope=(cudaScene->viewList[idGroup].center[2]- ((float*)xyOffset)[2])/
			sqrt(getPlaneDis((float*)xyOffset,cudaScene->viewList[idGroup].center))*FLOAT_INT_RATE;
		//如果超过了所在分区的阈值，则认为需要找一个新的图片
		if(mainSlope<directionSlope[mainSegment]*2)
		{
			//当遇到遮挡问题时，还是继续由第1个线程来处理，频繁同步未必就快
			//最小的投影角度
			unsigned minSlope=mainSlope;
			unsigned newIdGroup=idGroup;
			for(int i=0;i<IMG_GROUP_SIZE;++i)
			{
				//判断当前方向的投影角度是否更合适 并且需要保证这个投影角度里是有相机可以使用的
				if(cudaScene->viewList[idGroup].imgGroup[i]<cudaScene->viewNum &&
					directionSlope[i]<minSlope)
				{
					//更新最合适的相机标号
					minSlope=directionSlope[i];
					newIdGroup=cudaScene->viewList[idGroup].imgGroup[i];
				}
			}
			//把最后得到的最佳角度更新到正式的idGroup上
			cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->idGroup=newIdGroup;
			idGroup=newIdGroup;
		}
		//计算完成后仍然是由第1个线程，计算当前位置的点在图片上的投影位置
		float projectLocal[2];
		getProjectLocal(cudaScene->viewList+idGroup,(float*)xyOffset,projectLocal);
		//在当前点的分数上记录投影位置
		uint16_t* cellProjectLocal=(uint16_t*)(&(cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->score));
		cellProjectLocal[0]=round(projectLocal[0]);
		cellProjectLocal[1]=round(projectLocal[1]);
	}
}

//对传入的cuda版的scene做染色处理，但这里面记录的是每个网格单元应该从哪个图片里面取颜色，以及取颜色的坐标
void colorScene(Scene* cudaScene,Scene* cpuScene)
{
	auto& gridInfo=cpuScene->grid->gridInfo;
	//处理每个图片的遮挡问题
	colorCells<<<dim3(gridInfo.gridSize[0],gridInfo.gridSize[1],1),
		dim3(IMG_GROUP_SIZE,IMG_GROUP_SIZE,1)>>>(cudaScene);
	cudaDeviceSynchronize();
}