#pragma once

//根据水平方向的向量计算相机落在的角度区间
__device__ uint8_t getAngleSeg(float* vec)
{
    //排除全是0的情况
    if(vec[0]==0 && vec[1]==0) return IMG_GROUP_SIZE+1;
    //计算极角
    float tempAngle=atan2(vec[1],vec[0])*180/M_PI;
    //极角加上22.5度，做区间平移
    tempAngle+=22.5;
    //确保在0~360度之间
    while(tempAngle>=360) tempAngle-=360;
    while(tempAngle<0) tempAngle+=360;
    //返回计算得到的区间
    return tempAngle/45;
}

//计算两个点的水平距离
//为了减少计算时间，这里返回的是平方，一般计算距离也只是为了比较，直接拿平方距离去比较就可以了
__device__ float getPlaneDis(float* point1,float* point2)
{
    return (point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]);
}

//计算点在相机中的投影位置
__device__ void getProjectLocal(ViewInfo* view,float* blockPoint,float* projectLocal)
{
    //把当前的点减去光心 这里目前是单线程运行的
    float centerOffset[3];
    for(int i=0;i<3;++i) centerOffset[i]=blockPoint[i]-view->center[i];
    //把减去光心的结果乘上带内参的旋转矩阵
    projectLocal[0]=view->rotation[0]*centerOffset[0]+view->rotation[1]*centerOffset[1]+view->rotation[2]*centerOffset[2];
    projectLocal[1]=view->rotation[3]*centerOffset[0]+view->rotation[4]*centerOffset[1]+view->rotation[5]*centerOffset[2];
    float homoItem=
        view->rotation[6]*centerOffset[0]+view->rotation[7]*centerOffset[1]+view->rotation[8]*centerOffset[2];
    //让投影位置的结果除以齐次项
    projectLocal[0]/=homoItem;
    projectLocal[1]/=homoItem;
}