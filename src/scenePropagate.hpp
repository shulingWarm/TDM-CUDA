#pragma once
#include"scene.hpp"
#include"userAtomic.hpp"
#include"config.hpp"
#include"cudaHandleError.hpp"
#include"windowConfig.hpp"
#include <cuda_runtime.h>
#include"grid3d.hpp"
#include"geometry.hpp"

#define WINDOWS_SIZE 3
#define REF_CELL_INIT 999999

//错误的投影数据，对于某个位置，如果它不存在投影向量，就把它记录成这个数字，表示这个投影颜色是非法的
#define INVALID_PIXEL -888888

//scene传播的总流程，涉及到scene里面的网格初始化之后的各种操作都会写在这里面

//准备每个相机的相机组的核函数
__global__ void kernelPrepareCameraGroup(Scene* cudaScene,float avgHeight)
{
	//当前线程负责的中心点
	__shared__ float blockPoint[3];
	//其它位置的相机到当前相机的中心的最近距离
	__shared__ float directionMinDis[IMG_GROUP_SIZE];
	//每个方向的分数
	__shared__ int directionMutex[IMG_GROUP_SIZE];
	//由第1线程初始化当前线程块的投影中心点
	if(threadIdx.x<2)
	{
		blockPoint[threadIdx.x] = cudaScene->viewList[blockIdx.x].center[threadIdx.x];
	}
	else if(threadIdx.x==2)
	{
		//z的情况记录的是平均高度
		blockPoint[2]=avgHeight;
	}
	//另外，把每个方向的最佳相机的标号设置为默认值
	else if(threadIdx.x<3+IMG_GROUP_SIZE)
	{
		cudaScene->viewList[blockIdx.x].imgGroup[threadIdx.x-3]=cudaScene->viewNum+1;
		//当前线程负责的每个方向的距离的最小值，把这个数据弄成一个很大的数字
		directionMinDis[threadIdx.x-3]=9999999;
		//把锁弄成打开的状态
		directionMutex[threadIdx.x-3]=0;
	}
	//同步
	__syncthreads();
	//和线程块标号相同的线程是不参数这里的计算的
	if(threadIdx.x!=blockIdx.x)
	{
		//只适用于当前线程的临时变量，表示block点在当前线程负责的图片上的投影位置
		float projectLocal[2];
		//block里面的点在当前相机中的投影位置
		getProjectLocal(cudaScene->viewList+(threadIdx.x),blockPoint,projectLocal);
		//判断这个相机的中心点能不能被当前线程负责的相机看到
		if(projectLocal[0]>0 && projectLocal[1]>0 && 
			projectLocal[0]<cudaScene->imgSize[0] && projectLocal[1]<cudaScene->imgSize[1])
		{
			//当前线程对应的光心
			float* threadCenter=cudaScene->viewList[threadIdx.x].center;
			//当前位置的相机到中心点的向量
			float planeVec[2]={threadCenter[0]- blockPoint[0], threadCenter[1] - blockPoint[1]};
			//计算两个点之间的水平距离
			projectLocal[0]=planeVec[0]*planeVec[0] + planeVec[1]*planeVec[1];
			//当前相机的点相对于当前位置的方向
			uint8_t angleSeg = getAngleSeg(planeVec);
			//需要保证得到的是一个有效区段
			if(angleSeg<IMG_GROUP_SIZE)
			{
				//根据点到目标的距离取最大值 
				atomicUpdateScore<float,uint16_t,0>(projectLocal[0],
					directionMinDis+angleSeg,threadIdx.x,cudaScene->viewList[blockIdx.x].imgGroup + angleSeg,
					directionMutex+angleSeg);
			}
		}
	}
}

//每个点到其它相机的距离描述
class CellDisToCamera
{
public:
	float planeDisToOthers;
	unsigned idView;//相机组的标号，虽然最开始是按照顺序来排列的，但后面会涉及到二分排序
};

//分配每个网格单元属于哪个相机组
__global__ void makeBestCameraGroup(Scene* cudaScene,
	Grid3D<CellDisToCamera>* gpuCellDisInfo)
{
	//当前的网格单元对应的水平位置
	__shared__ float planeLocal[2];
	//共享内存，当前网格单元到其它的相机中心的水平距离
	__shared__ CellDisToCamera* cellDisInfo;
	//二分查找的遍历范围
	__shared__ unsigned iterateRange;
	//由第1个线程初始化共享内存
	if(threadIdx.x==0)
	{
		iterateRange=(cudaScene->viewNum+1)/2;
	}
	else if(threadIdx.x==1)
	{
		//初始化当前点的水平位置
		planeLocal[0]=cudaScene->grid->gridInfo.toWorldCoord(0,blockIdx.x);
		planeLocal[1]=cudaScene->grid->gridInfo.toWorldCoord(1,blockIdx.y);
	}
	else if(threadIdx.x==2)
	{
		//从全局内存里面获取属于本线程块的中间变量
		cellDisInfo=gpuCellDisInfo->getDataAt(blockIdx.x,blockIdx.y,0);
	}
	__syncthreads();
	//计算当前线程负责的相机到planeLocal的距离
	cellDisInfo[threadIdx.x].planeDisToOthers=getPlaneDis(planeLocal,cudaScene->viewList[threadIdx.x].center);
	//把标号保存成当前线程的标号
	cellDisInfo[threadIdx.x].idView=threadIdx.x;
	__syncthreads();
	//循环进行二分选择最优
	while(iterateRange>0)
	{
		//只有在遍历范围内的才需要处理，并且需要保证比较的位置也没有超界
		if(threadIdx.x<iterateRange && threadIdx.x+iterateRange<cudaScene->viewNum)
		{
			//判断比较位置的距离是否比当前的距离更小
			if(cellDisInfo[threadIdx.x + iterateRange].planeDisToOthers < 
				cellDisInfo[threadIdx.x].planeDisToOthers)
			{
				//把比较位置的网格单元信息更新到当前的位置
				cellDisInfo[threadIdx.x]=cellDisInfo[threadIdx.x + iterateRange];
			}
		}
		__syncthreads();
		//由第1个线程更新下一个周期的迭代信息
		if(threadIdx.x==0)
		{
			//如果已经是1了，直接更新成0
			if(iterateRange==1)
			{
				iterateRange=0;
			}
			else
			{
				iterateRange=(iterateRange+1)/2;
			}
		}
		__syncthreads();
	}
	//把最后得到的最佳相机组更新到view信息里面
	if(threadIdx.x==0)
	{
		cudaScene->grid->getDataAt(blockIdx.x,blockIdx.y)->idGroup=cellDisInfo[0].idView;
	}
}

//准备每个相机的相机组
void prepareCameraGroup(Scene* cpuScene,
	Scene* cudaScene,float avgHeight)
{
	//调用核函数
	kernelPrepareCameraGroup<<<cpuScene->viewNum,cpuScene->viewNum>>>(cudaScene,avgHeight);
	//需要准备一个三维网格，同样需要准备一个getDataAt 这只是一个中间变量
	Grid3D<CellDisToCamera> cpuCellDisInfo;
	cpuCellDisInfo.cudaInit(cpuScene->grid->gridInfo.gridSize[0],cpuScene->grid->gridInfo.gridSize[1],
		cpuScene->viewNum);
	//cuda形式的三维网格
	Grid3D<CellDisToCamera>* gpuCellDisInfo=cpuCellDisInfo.toCuda();
	//准备每个网格单元属于哪个相机组
	makeBestCameraGroup<<<dim3(cpuScene->grid->gridInfo.gridSize[0],
		cpuScene->grid->gridInfo.gridSize[1],1),cpuScene->viewNum>>>(cudaScene,gpuCellDisInfo);
	//释放三维网格
	handleError(cudaFree(cpuCellDisInfo.data));
	handleError(cudaFree(gpuCellDisInfo));
}

//一个图上的颜色向量的信息
class ImageColorVector
{
public:
	//图片上的9个点
	float data[WINDOWS_SIZE*WINDOWS_SIZE];
};

//滑窗时的运行时信息
//其实是个运行时信息的大杂绘
class SlideInfo
{
public:
	//参考网格单元的位置 x,y的变化范围只是0~2 表示这9个网格单元里面谁是参考网格单元
	uint16_t refCell[2];
	//当前的block的左上角对应的网格坐标
	uint16_t blockCornerLocal[2];
	//xy方向随机变化的数值
	float xySlope[2];

	//9个点在每个图上的颜色向量 最后一位是相机组的中心相机的颜色向量
	ImageColorVector viewColorVectors[IMG_GROUP_SIZE + 1];

	//根据输入的3*3xy获取对应的网格域内的xy坐标
	__device__ int getBlockXy(char yFlag,int x)
	{
		return blockCornerLocal[yFlag] + x;
	}

	//根据传入的网格信息和x,y的位置，获取属于当前xy位置的z值
	__device__ float getHeightByThreadXy(int x,int y,Grid<DOMCell>* grid)
	{
		return grid->getDataAt(blockCornerLocal[0]+refCell[0],blockCornerLocal[1]+refCell[1])->z_ +
			xySlope[0]*(x- refCell[0]) + xySlope[1]*(y - refCell[1]);
	}

	//根据传入的x,y的位置，获取当前对应的实际网格单元
	__device__ DOMCell* getDataAt(int x,int y,Grid<DOMCell>* grid)
	{
		return grid->getDataAt(blockCornerLocal[0]+x,blockCornerLocal[1]+y);
	}

	//当前窗口里面的参考单元是否可用，也就是看一下它的参考网格单元是否被初始化过
	__device__ char isRefCellUsable()
	{
		return ((int*)refCell)[0]!=REF_CELL_INIT;
	}
};

//传入新的参考位置的坐标，把它更新到指针上，原子操作
__device__ void updateRefCell(uint16_t x,uint16_t y,int* refCell)
{
	//初始化用于更新数据的int变量
	int newXy;
	((uint16_t*)(&newXy))[0]=x;
	((uint16_t*)(&newXy))[1]=y;
	//只有当目标位置为0的时候才修改它
	atomicCAS(refCell,REF_CELL_INIT,newXy);
}

//寻找可参考的网格单元
__device__ void findReferenceCell(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	SlideInfo* slideShare)
{
	//只由每个位置的第1个线程来寻找它是否需要被参考
	if(threadIdx.z==0)
	{
		//当前线程负责的网格单元
		DOMCell* threadData=cudaScene->grid->getDataAt(slideShare->blockCornerLocal[0]+threadIdx.x,
			slideShare->blockCornerLocal[1]+threadIdx.y);
		//判断自己当前负责的网格单元的分数是否可以用来被参考
		if(cudaWindowConfig->isScoreUsable(threadData->score) &&
			threadData->refTime > 0 //确保当前的网格单元还有可被参考的次数
		)
		{
			//在可参考位置上更新当前的坐标
			updateRefCell(threadIdx.x,threadIdx.y,(int*)(slideShare->refCell));
		}
	}
	__syncthreads();
}

//计算每个线程的颜色向量
__device__ void makeColorVectors(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	SlideInfo* slideShare)
{
	//当前线程负责的view标号
	unsigned idView=slideShare->getDataAt(slideShare->refCell[0],slideShare->refCell[1],cudaScene->grid)->idGroup;
	//需要处理的目标颜色
	float* dstPixel=slideShare->viewColorVectors[threadIdx.z].data + (threadIdx.y*WINDOWS_SIZE + threadIdx.x);
	//对于z线程的最后一块，它拿到的就是线程块本身 但z线程的其它位置却需要把自己更新成相机组里面的对应位置
	if(threadIdx.z<IMG_GROUP_SIZE)
	{
		idView=cudaScene->viewList[idView].imgGroup[threadIdx.z];
	}
	//如果idView是不合法的,就不需要计算了，把对应位置的颜色保存成一个很离谱的数据就可以了
	if(idView>=cudaScene->viewNum)
	{
		dstPixel[0]=INVALID_PIXEL;
	}
	else
	{
		//当前位置坐标对应的三维点
		float point3d[3]={cudaScene->grid->gridInfo.toWorldCoord(0,slideShare->getBlockXy(0,threadIdx.x)),
			cudaScene->grid->gridInfo.toWorldCoord(1,slideShare->getBlockXy(1,threadIdx.y)),
			slideShare->getHeightByThreadXy(threadIdx.x,threadIdx.y,cudaScene->grid)};
		//初始化投影点的位置
		float projectLocal[2]={1,2};
		//对于当前的点，把点投影到对应的图片上
		getProjectLocal(cudaScene->viewList+idView,point3d,projectLocal);
		dstPixel[0]=tex2D<uchar>(cudaScene->viewList[idView].texObj,projectLocal[0],projectLocal[1]);
		//如果得到的颜色是非法的，就把它记录成一个更离谱的颜色
		if(dstPixel[0]<0.5)
		{
			dstPixel[0]=INVALID_PIXEL;
		}
	}
}

//计算平均颜色向量时用到的中间值
class AverageColorMid
{
public:
	//每一行的颜色向量的平均值
	float rowAverage[3];
};

//所有的颜色向量减去自己的均值
__device__ void colorVectorMinusAverage(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	SlideInfo* slideShare)
{
	//每个图片上的颜色平均值
	__shared__ AverageColorMid avgMid[IMG_GROUP_SIZE+1];
	//计算每个图片上的颜色向量的平均值
	if(threadIdx.x==0)
	{
		avgMid[threadIdx.z].rowAverage[threadIdx.y]=0;
		//把每个行的线程加到自己负责的这一行上
		for(int i=0;i<WINDOWS_SIZE;++i) avgMid[threadIdx.z].rowAverage[threadIdx.y]+=
			slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + i];
	}
	__syncthreads();
	//已经计算得到了每一行的求和结果，下一步计算每个颜色向量总的结果
	if(threadIdx.x==0 && threadIdx.y==0)
	{
		avgMid[threadIdx.z].rowAverage[0]+=avgMid[threadIdx.z].rowAverage[1];
		avgMid[threadIdx.z].rowAverage[0]+=avgMid[threadIdx.z].rowAverage[2];
		//如果得到的是一个非法的数据，就进一步把它标注成非法的数据
		if(avgMid[threadIdx.z].rowAverage[0]<0)
		{
			//把它标注成非法数据
			avgMid[threadIdx.z].rowAverage[0]=INVALID_PIXEL;
		}
		else
		{
			avgMid[threadIdx.z].rowAverage[0]/=(WINDOWS_SIZE*WINDOWS_SIZE);
		}
	}
	__syncthreads();
	//判断自己的平均颜色是否合法，如果不合法就把自己也记录成不合法的颜色
	if(avgMid[threadIdx.z].rowAverage[0]<0)
	{
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + threadIdx.x]=INVALID_PIXEL;
	}
	else
	{
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + threadIdx.x]-=
			avgMid[threadIdx.z].rowAverage[0];
	}
	__syncthreads();
	//直接在rowAverage上计算减去均值后结果的平方和
	if(avgMid[threadIdx.z].rowAverage[0]>0 && threadIdx.x==0)
	{
		avgMid[threadIdx.z].rowAverage[threadIdx.y]=0;
		//把每个行的线程加到自己负责的这一行上
		for(int i=0;i<WINDOWS_SIZE;++i) avgMid[threadIdx.z].rowAverage[threadIdx.y]+=
			(slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + i]*
				slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + i]);
	}
	__syncthreads();
	//上面是计算每一行的平方和的结果，下在要把每一行的平方和结果加起来，形成总的平方和
	if(avgMid[threadIdx.z].rowAverage[0]>0 && threadIdx.x==0 && threadIdx.y==0)
	{
		//把每一行的平方和加起来
		avgMid[threadIdx.z].rowAverage[0]+=avgMid[threadIdx.z].rowAverage[1];
		avgMid[threadIdx.z].rowAverage[0]+=avgMid[threadIdx.z].rowAverage[2];
		//这里不需要再考虑无效像素的问题了，无效像素根据走不进这个分支
		avgMid[threadIdx.z].rowAverage[0]=sqrt(avgMid[threadIdx.z].rowAverage[0]);
	}
	__syncthreads();
	//对所有的颜色做归一化处理 但只有合法的颜色像素才参与计算
	if(avgMid[threadIdx.z].rowAverage[0]>(INVALID_PIXEL + 1))
	{
		//所有的颜色向量除以平方和的均值
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE + threadIdx.x]/=
			avgMid[threadIdx.z].rowAverage[0];
	}
	__syncthreads();
}

//计算颜色向量的分数，最后会把分数保存在slideInfo里面
__device__ void getColorVectorScore(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	SlideInfo* slideShare)
{
	//求点积结果的和的时候会用到，在这里先初始化了
	__shared__ unsigned iterateRange;
	//把每个颜色向量和参考颜色的向量相乘 但首先需要确保颜色向量是有效的
	if(threadIdx.z<IMG_GROUP_SIZE && 
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y * WINDOWS_SIZE + threadIdx.x]>(INVALID_PIXEL + 1))
	{
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y * WINDOWS_SIZE + threadIdx.x]*=
			slideShare->viewColorVectors[IMG_GROUP_SIZE].data[threadIdx.y * WINDOWS_SIZE + threadIdx.x];
	}
	else if(threadIdx.z==IMG_GROUP_SIZE && threadIdx.x==0 && threadIdx.y==0)
	{
		iterateRange=(IMG_GROUP_SIZE+1)/2;
	}
	__syncthreads();
	//把点乘的结果保存在每个向量的第1位，这个时候已经可以破坏原有的向量了
	//参考颜色向量不需要进行这样的计算
	if(threadIdx.x==0 && threadIdx.z<IMG_GROUP_SIZE && threadIdx.z<IMG_GROUP_SIZE && 
		slideShare->viewColorVectors[threadIdx.z].data[0]>(INVALID_PIXEL + 1))
	{
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE]+=
			slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE+1];
		slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE]+=
			slideShare->viewColorVectors[threadIdx.z].data[threadIdx.y*WINDOWS_SIZE+2];
	}
	__syncthreads();
	//上面是把每一行的颜色向量加起来，下一步是把每一行的求和结果加起来，形成点积的最终结果
	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z<IMG_GROUP_SIZE)
	{
		//颜色向量是否合法需要单独分情况处理
		if(slideShare->viewColorVectors[threadIdx.z].data[0]> (INVALID_PIXEL + 1))
		{
			slideShare->viewColorVectors[threadIdx.z].data[0]+=slideShare->viewColorVectors[threadIdx.z].data[WINDOWS_SIZE];
			slideShare->viewColorVectors[threadIdx.z].data[0]+=slideShare->viewColorVectors[threadIdx.z].data[WINDOWS_SIZE*2];
			//记录当前颜色向量的求和结果是由几个图片组成的 对于非法的颜色，这个位置是0,正常情况下这个位置刚开始是1
			slideShare->viewColorVectors[threadIdx.z].data[1]=1;
		}
		else
		{
			//对于非法的颜色向量，这个位置是0
			slideShare->viewColorVectors[threadIdx.z].data[1]=0;
		}
	}
	__syncthreads();
	//把每个图片上的点积结果加起来，加到第1个图片上
	while(iterateRange>0)
	{
		//由于只是各个图片上的维度操作，每个图片只出一个坐标位置就可以了
		if(threadIdx.x==0 && threadIdx.y==0)
		{
			//查看z维度的线程是否在遍历的范围内
			if(threadIdx.z<iterateRange && threadIdx.z+iterateRange<IMG_GROUP_SIZE)
			{
				//如果比较位置具有有效的向量再作处理
				if(slideShare->viewColorVectors[threadIdx.z+iterateRange].data[1]>0.5)
				{
					//如果当前位置原本没有向量，就直接覆盖
					if(slideShare->viewColorVectors[threadIdx.z].data[1]<0.5)
					{
						slideShare->viewColorVectors[threadIdx.z].data[0]=
							slideShare->viewColorVectors[threadIdx.z+iterateRange].data[0];
						slideShare->viewColorVectors[threadIdx.z].data[1]=
							slideShare->viewColorVectors[threadIdx.z+iterateRange].data[1];
					}
					else//当前位置的向量有效的情况下把两个位置的向量加起来
					{
						slideShare->viewColorVectors[threadIdx.z].data[0]+=
							slideShare->viewColorVectors[threadIdx.z+iterateRange].data[0];
						slideShare->viewColorVectors[threadIdx.z].data[1]+=
							slideShare->viewColorVectors[threadIdx.z+iterateRange].data[1];
					}
				}
			}
		}
		__syncthreads();
		//准备下一个周期的迭代信息
		if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
		{
			//如果迭代周期已经是1了说明下一个周期就可以结束了
			if(iterateRange==1)
			{
				iterateRange=0;
			}
			else
			{
				//更新到下一个迭代周期
				iterateRange=(iterateRange+1)/2;
			}
		}
		__syncthreads();
	}
	//把分数保存在颜色向量的第1个位置上
	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 && slideShare->viewColorVectors[0].data[1]>0.5)
	{
		slideShare->viewColorVectors[0].data[0]/=slideShare->viewColorVectors[0].data[1];
		//如果分数小于0,就把它记录成0 使用float分数的情况下不需要做下面的操作
		// if(slideShare->viewColorVectors[0].data[0]<0)
		// {
		// 	slideShare->viewColorVectors[0].data[0]=0;
		// }
		// else
		// {
		// 	//否则要把这个分数变换到char的数据类型范围下，因为这里为了节省空间，每个网格单元的分数其实是用char来存储的
		// 	slideShare->viewColorVectors[0].data[0]*=255;
		// }
	}
	__syncthreads();
}

//更新每个网格单元的分数
__device__ void updateCellScore(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	SlideInfo* slideShare)
{
	//更新算出来的分数 8个边图呢起码要有3个是能看到的吧
	if(threadIdx.z==0 && slideShare->viewColorVectors[0].data[1]>3)
	{
		//判断算出来的分数是否高于自己控制的网格单元的分数
		if(slideShare->viewColorVectors[0].data[0] > slideShare->getDataAt(threadIdx.x,threadIdx.y,cudaScene->grid)->score)
		{
			//更新分数
			slideShare->getDataAt(threadIdx.x,threadIdx.y,cudaScene->grid)->score=slideShare->viewColorVectors[0].data[0];
			//更新高度
			slideShare->getDataAt(threadIdx.x,threadIdx.y,cudaScene->grid)->z_=slideShare->getHeightByThreadXy(
				threadIdx.x,threadIdx.y,cudaScene->grid);
			//当一个网格单元有了新的高度，就给它一次新的水平传播的机会
			slideShare->getDataAt(threadIdx.x,threadIdx.y,cudaScene->grid)->refTime=INIT_REF;
			//所属相机组也需要跟随一下
			if(threadIdx.x!=slideShare->refCell[0] && threadIdx.y!=slideShare->refCell[1])
			{
				slideShare->getDataAt(threadIdx.x,threadIdx.y,cudaScene->grid)->idGroup=
					slideShare->getDataAt(slideShare->refCell[0],slideShare->refCell[1],cudaScene->grid)->idGroup;
			}
		}
	}
}

//滑窗过程的核函数
__global__ void windowSlideKernel(Scene* cudaScene,WindowConfig* cudaWindowConfig,
	char idOffset //虽然windowConfig里面也有idOffset,但那个只是用来计算周期的，使用那个地方的idOffset可能会涉及到同步问题
)
{
	//运行时用到的相关信息
	__shared__ SlideInfo slideShare;
	//由第1个线程负责初始化共享内存
	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
	{
		//把参考的cell弄成奇怪的数
		((int*)(slideShare.refCell))[0]=REF_CELL_INIT;
	}
	else if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==1)
	{
		//初始化当前线程块在左上角的位置
		slideShare.blockCornerLocal[0]=blockIdx.x*WINDOWS_SIZE + idOffset;
		slideShare.blockCornerLocal[1]=blockIdx.y*WINDOWS_SIZE + idOffset;
	}
	__syncthreads();
	//从9个网格单元里面寻找可参考的网格单元
	findReferenceCell(cudaScene,cudaWindowConfig,&slideShare);
	//根据参考网格单元的剩余次数初始化法向量 首先要保证找到了一个可参考的网格单元
	if(slideShare.isRefCellUsable() &&
	 threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
	{
		//参考网格单元
		DOMCell* refCell=cudaScene->grid->getDataAt(
			slideShare.refCell[0]+slideShare.blockCornerLocal[0],
			slideShare.refCell[1]+slideShare.blockCornerLocal[1]);
		//判断当前选中的相机块是不是第1次被使用
		if(refCell->refTime==INIT_REF)
		{
			//把倾斜角度直接弄成0,不倾斜
			slideShare.xySlope[0]=0;
			slideShare.xySlope[1]=0;
		}
		else
		{
			//用两个随机量初始化斜率 变化范围是-2~2
			slideShare.xySlope[0]=curand_uniform(&(refCell->randState))*4-2;
			slideShare.xySlope[1]=curand_uniform(&(refCell->randState))*4-2;
			//这样得到的斜率是空间域上的斜率，需要乘上空间分辨率把它弄成网格域上的斜率
			slideShare.xySlope[0]*=cudaScene->grid->gridInfo.pixelResolution;
			slideShare.xySlope[1]*=cudaScene->grid->gridInfo.pixelResolution;
		}
		//把被选中的参考网格单元的可参考次数减小
		--refCell->refTime;
	}
	__syncthreads();
	//只有找到了可参考的网格单元才继续向下走 这里是全线程参与
	if(slideShare.isRefCellUsable())
	{
		//准备每个线程的投影颜色
		makeColorVectors(cudaScene,cudaWindowConfig,&slideShare);
		__syncthreads();
		//每个找到的颜色减去自己的均值
		colorVectorMinusAverage(cudaScene,cudaWindowConfig,&slideShare);
		//计算颜色向量的分数 只有参考颜色向量有效才会进入这个分支
		if(slideShare.viewColorVectors[IMG_GROUP_SIZE].data[0]>-0.5)
		{
			getColorVectorScore(cudaScene,cudaWindowConfig,&slideShare);
			//根据算出来的分数结果，更新每个网格单元的分数
			updateCellScore(cudaScene,cudaWindowConfig,&slideShare);
		}
	}
	//由第1个block负责更新每个周期的运行选项 这里不需要考虑同步问题，可以任由它发生访问冲突问题
	//访问冲突了影响不大
	if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
	{
		cudaWindowConfig->iterateNext();
	}
}

//根据偏移量计算本次迭代时应当采取的线程块大小
//目前窗口和大小就默认是3
dim3 getSlideKernelSize(unsigned gridWidth,
	unsigned gridHeight,
	unsigned idOffset
)
{
	const unsigned windowSize=3;
	return dim3((gridWidth- idOffset)/windowSize,(gridHeight- idOffset)/windowSize,1);
}

void scenePropagate(Scene* cpuScene, //传进来一个cpu的scene也只是为了方便读取scene里面的网格大小
	Scene* cudaScene,
	float avgHeight //场景的平均高度
)
{	//制作基本的相机组，即每个相机组里面都有哪些相机
	prepareCameraGroup(cpuScene,cudaScene,avgHeight);
	//滑窗传播时用到的配置信息
	WindowConfig cpuWindowConfig;
	//把数据转到cuda
	WindowConfig* cudaWindowConfig=cpuWindowConfig.toCuda();
	//准备开始进行滑窗传播
	for(unsigned idIterate=0;idIterate<cpuWindowConfig.maxIterate;++idIterate)
	{
		//遍历窗口的三个偏移量
		for(unsigned idOffset=0;idOffset<3;++idOffset)
		{
			//启动核函数
			windowSlideKernel<<<getSlideKernelSize(
				cpuScene->grid->gridInfo.gridSize[0],cpuScene->grid->gridInfo.gridSize[1],idOffset),dim3(3,3,9)>>>(
				cudaScene,cudaWindowConfig,idOffset);
		}
	}
	cudaDeviceSynchronize();
	//释放cuda版本的窗口处理
	handleError(cudaFree(cudaWindowConfig));
}