#pragma once
#include"cudaHandleError.hpp"

//三维的通用网格，但暂时不涉及到分辨率的问题
//指针的排列顺序是y,x,z y相同的指针是连续的，(x,y)相同的指针是连续的
template<class VoxelType>
class Grid3D
{
public:
	//原始的数据
	VoxelType* data;

	//网格的大小
	unsigned gridSize[3];

	//获取网格数据的总大小
	unsigned getDataSize()
	{
		return gridSize[0]*gridSize[1]*gridSize[2];
	}

	//初始化网格单元
	//但是直接把数据开辟在cuda上
	void cudaInit(unsigned gridWidth,unsigned gridHeight,unsigned gridChannel)
	{
		//复制xyz的范围
		gridSize[0]=gridWidth;
		gridSize[1]=gridHeight;
		gridSize[2]=gridChannel;
		//给data开辟空间
		handleError(cudaMalloc((void**)&data,sizeof(VoxelType)*getDataSize()));
	}

	//获取三维的数据
	__device__ __host__ VoxelType* getDataAt(unsigned x,unsigned y,unsigned z)
	{
		return data+(y*gridSize[0]*gridSize[2] + x*gridSize[2] + z);
	}

	//把当前的类转换到cuda上
	//这里暂时是直接初始化在cuda上的，还没有做初始化在cpu上的功能，主要是目前还不需要
	Grid3D<VoxelType>* toCuda()
	{
		//新建临时的cuda指针
		Grid3D<VoxelType>* cudaGrid;
		//开辟cuda的空间
		handleError(cudaMalloc((void**)&cudaGrid,sizeof(Grid3D<VoxelType>)));
		//把当前类内的内容复制到cuda内存里面，data已经是cuda的状态了
		handleError(cudaMemcpy(cudaGrid,this,sizeof(Grid3D<VoxelType>),cudaMemcpyHostToDevice));
		return cudaGrid;
	}
};