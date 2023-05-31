#pragma once

//适用于float的max原子操作
__device__ void floatAtomicMax(float* dstData,float cmpData)
{
	while(true)
	{
		//测试性地比较
		float tempReadAns=*dstData;
		int* tempAddress=(int*)(&tempReadAns);
		//比较数据
		if(cmpData>tempReadAns)
		{
			//如果修改成功就结束 先判断一下dstData是否还等于刚才看到的那个值，如果等于就继续修改
            if(atomicCAS((int*)dstData,*tempAddress,*((int*)(&cmpData)))==*tempAddress)
            {
                break;
            }
		}
		else //比较失败的情况下也结束
		{
			break;
		}
	}
}

//涉及到分数比较的操作，传入分数1和标号1,另外传入两个标号，如果一个分数大于另一个分数，则更新分数
template<typename Score,typename Identity,int saveBig>
__device__ void atomicUpdateScore(Score newScore,Score* bestScore,
	Identity newId,Identity* bestId,int* mutex)
{
	//等待计数
	int waitCount=10000;
	//给mutex上锁
	while(atomicCAS(mutex,0,newId+1)!=(newId+1) && waitCount>0)
	{
		--waitCount;
	}
	//更新分数
	if((saveBig && newScore>bestScore[0]) || (saveBig==0 && newScore<bestScore[0]))
	{
		bestId[0]=newId;
		bestScore[0]=newScore;
	}
	//解锁
	atomicExch(mutex,0);
}