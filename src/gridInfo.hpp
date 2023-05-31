#pragma once

//通用的网格信息，特指的是那种涉及到离散和连续之间映射的那种网格
class GridInfo
{
public:
	//网格单元的分辨率
	float pixelResolution;
	//网格单元的x,y最小值
	float xyMin[2];
	//网格单元的长和宽
	unsigned gridSize[2];

	//获取data数据的总长度
	__device__ __host__ unsigned getDataSize()
	{
		return gridSize[0]*gridSize[1];
	}

	//判断一个数据是否在网格的范围内
	__device__ __host__ char isPointInRange(int* xy)
	{
		return xy[0]>=0 && xy[0]<gridSize[0] && 
			xy[1]>=0 && xy[1]<gridSize[1];
	}

	//转换成离散坐标
	//传入0的时候是对x做处理，传1的时候是把传入的数值理解成y
	__device__ __host__ int toGridCoord(char yFlag,float x)
	{
		return (x- xyMin[yFlag])/pixelResolution;
	}

	//把离散点的坐标转换到世界坐标系
	__device__ __host__ float toWorldCoord(char yFlag,int x)
	{
		return (x+0.5f)*pixelResolution + xyMin[yFlag];
	}

	//转换成cuda的函数
	GridInfo* toCuda()
	{
		//初始化一个用于cuda内存的指针
		GridInfo* cudaInfo;
		cudaMalloc((void**)&cudaInfo,sizeof(GridInfo));
		//把当前的内容复制到cuda内存中
		cudaMemcpy(cudaInfo,this,sizeof(GridInfo),cudaMemcpyHostToDevice);
		return cudaInfo;
	}

	//空的构造函数
	GridInfo(){}

	//用范围和分辨率初始化
	void init(float* domRange,float pixelLength)
	{
		//记录分辨率
		this->pixelResolution=pixelLength;
		//计算xy的最小值
		xyMin[0]=domRange[0];
		xyMin[1]=domRange[2];
		//计算宽和高
		gridSize[0]=(domRange[1]- domRange[0])/pixelLength;
		gridSize[1]=(domRange[3]- domRange[2])/pixelLength;
	}

	//传入范围和分辨率的构造函数
	GridInfo(float* domRange,float pixelLength)
	{
		//调用网格信息的初始化
		init(domRange,pixelLength);
	}
};