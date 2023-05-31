#pragma once
#include"config.hpp"
#include <curand_kernel.h>//使用cuda中的随机数需要用到这个头文件

#define INIT_REF 40

//其实就是一个struct,
class DOMCell
{
public:
	//当前网格单元的z值
	float z_=-1;
	curandState_t randState;//当前网格单元里面的随机数状态 为了并行处理，每个网格单元里面都保留了一个随机数状态
	//当前网格单元所属的相机组，其实是一个价值不大的变量，有点浪费内存
	uint16_t idGroup=0;
	//每个网格单元剩余的可参考次数
	uint8_t refTime=INIT_REF;
	//当前网格单元的分数
	ScoreType score=0;
};