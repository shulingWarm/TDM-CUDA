#pragma once
#include"gridInfo.hpp"
#include<stdlib.h>
#include<vector>
#include"cudaHandleError.hpp"

//通用的网格
//注意GridCell里面不要使用动态大小的数据类型
template<class GridCell>
class Grid
{
public:
	//网格单元的大小
	GridInfo gridInfo;

	//网格的数据
	GridCell* data=nullptr;

	//获取目标位置的数据
	__device__ __host__ GridCell* getDataAt(int x,int y)
	{
		return data + (y*gridInfo.gridSize[0] + x);
	}

	//把数据转换成cuda
	Grid<GridCell>* toCuda()
	{
		//临时新建一个用于返回的变量
		Grid<GridCell>* cudaGrid;
		handleError(cudaMalloc((void**)&cudaGrid,sizeof(Grid<GridCell>)));
		//新建一个临时的变量cpu变量，需要先把自己的成员变量弄成gpu的形式
		auto tempData=*this;
		//把data换成cuda的形式
		handleError(cudaMalloc((void**)(&tempData.data),sizeof(GridCell)*gridInfo.getDataSize()));
		handleError( cudaMemcpy(tempData.data,this->data,sizeof(GridCell)*gridInfo.getDataSize(),cudaMemcpyHostToDevice));
		//把临时数据里面的内容转换成cuda的形式
		handleError( cudaMemcpy(cudaGrid,&tempData,sizeof(*this),cudaMemcpyHostToDevice));
		return cudaGrid;
	}

	//用一个cuda数据初始化当前的数据 这个函数目前是很不通用的，不会检测数据是否正常开辟过
	void fromCuda(Grid<GridCell>* cudaGrid)
	{
		//先复制一个临时的grid
		Grid<GridCell> tempGrid;
		handleError(cudaMemcpy(&tempGrid,cudaGrid,sizeof(Grid<GridCell>),cudaMemcpyDeviceToHost));
		//复制数据里面的data
		handleError(cudaMemcpy(data,tempGrid.data,sizeof(GridCell)*gridInfo.getDataSize(),cudaMemcpyDeviceToHost));
		//释放数据里面的data
		handleError(cudaFree(tempGrid.data));
		//释放cuda数据的本体
		handleError(cudaFree(cudaGrid));
	}

	//初始化网格的函数
	void init(GridInfo gridInfo)
	{
		//初始化类内的成员
		this->gridInfo=gridInfo;
		//初始化网格数据的大小
		data=(GridCell*)malloc(sizeof(GridCell)*gridInfo.getDataSize());
	}

	Grid(){}

	Grid(GridInfo gridInfo){
		this->init(gridInfo);
	}

	//释放网格单元的数据
	void release()
	{
		//查看是否需要释放网格单元的数据
		if(data)
		{
			free(data);
			data=nullptr;
		}
	}
};