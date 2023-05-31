#pragma once
#include"config.hpp"

//滑窗传播时涉及到的配置信息
class WindowConfig
{
public:
	//总的目标迭代次数
	const unsigned maxIterate=2500;
	//当前的偏移量 //在迭代过程中的变化也就是0 1 2
	char idOffset=0;
	//期望的阈值分数
	ScoreType scoreThreshold=0.8;
	//可参考的分数偏移量
	ScoreType scoreOffset=0.2;
	//阈值分数每次下降的个数
	ScoreType scoreDropStep=0.001;
	//当前的迭代次数
	unsigned idIterate=0;
	//阈值分数从什么时候下降
	unsigned beginDropId=2000;

	//把当前的数据转到cuda
	WindowConfig* toCuda()
	{
		//新建临时的gpu版本的cuda
		WindowConfig* gpuWindowConfig;
		cudaMalloc((void**)&gpuWindowConfig,sizeof(WindowConfig));
		//直接把当前的数据复制给cuda
		cudaMemcpy(gpuWindowConfig,this,sizeof(WindowConfig),cudaMemcpyHostToDevice);
		//返回转到cuda的指针
		return gpuWindowConfig;
	}

	//把迭代次数更新到下一周期
	__device__ __host__ void iterateNext()
	{
		//判断偏移量是否已经到了2
		if(idOffset>=2)
		{
			idOffset=0;
			++idIterate;
			//如果迭代次数到了一定的程度，会开始下降阈值分数
			if(idIterate>beginDropId)
			{
				//减少一下期望的阈值分数 float的情况下直接减就可以，不用弄得那么精确
				scoreThreshold-=scoreDropStep;
			}
		}
		else
		{
			++idOffset;
		}
	}

	//判断一个传入的分数是否可以用来被参考
	__device__ char isScoreUsable(ScoreType score)
	{
		return score>scoreThreshold- scoreOffset;
	}
};