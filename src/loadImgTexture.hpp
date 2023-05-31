#pragma once

#include<opencv2/opencv.hpp>
#include<string>

typedef unsigned char uchar;

//把指定图片载入到某个纹理内存对象里面
void loadImgObjArray(const std::string& imgPath,cudaTextureObject_t* dstObj,
	cudaArray_t* dstArray)
{
	//新建一个texObj用于程序临时处理，处理完成之后把它保存到dstObj里面
	cudaTextureObject_t texObj;
	cudaArray_t cuArray;
	//读取灰度图
	cv::Mat srcImg=cv::imread(imgPath,cv::IMREAD_GRAYSCALE);
	if(srcImg.empty())
	{
		std::cerr<<"cannot load "<<imgPath<<std::endl;
		return;
	}
	//准备记录中间过程的运行效果
	cudaError_t statu;
	//新建cuda的通道描述符 这句表示只有一个通道，这个通道是8位的，其实就是灰度图
	cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	//给array分配描述符和它的长宽
	statu=cudaMallocArray(&cuArray, &channelDesc, srcImg.cols, srcImg.rows);
	if(statu!=cudaSuccess) std::cerr<<cudaGetErrorString(statu)<<std::endl;
	//把数据复制到array里面
    statu= cudaMemcpy2DToArray(cuArray, 
    	0, //行偏移量
    	0, //列偏移量
    	srcImg.data, //原始数据
    	srcImg.cols*sizeof(uchar), //每一行的字节数 这里指的是原始数据的每一行的字节宽度
    	srcImg.cols*sizeof(uchar),//需要复制的数据的宽度 以字节为单位
        srcImg.rows,//要复制的行数
    	cudaMemcpyHostToDevice);
	if(statu!=cudaSuccess) std::cerr<<cudaGetErrorString(statu)<<std::endl;
	//新建数据资源的描述符
    cudaResourceDesc resDesc;
    //这个数据一般都需要被初始化成0
    memset(&resDesc,0,sizeof(resDesc));
    //指定数据的资源类型是array
    resDesc.resType = cudaResourceTypeArray;
    //指定数组资源的位置
    resDesc.res.array.array = cuArray;
    //新建纹理描述符
    cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(texDesc));
    //设置xy方向访问越界时的操作，返回一个固定的颜色值，黑色
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    //当超过访问的边界时，返回黑色
    for(int i=0;i<4;++i) texDesc.borderColor[i]=-500;
	//如果设置成cudaFilterModePoint则表示最近点 cudaFilterModeLinear表示双线性插值访问
    texDesc.filterMode = cudaFilterModePoint;
	//把读取的数据就按照指定的类型来处理，得到的是一个float4的数据类型
	//如果设置成cudaReadModeNormalizedFloat则会把数据处理成0~1的浮点数，也就是归一化的浮点数
	//如果设置成cudaReadModeRaw则是直接返回数据的二进制原始数据，不对数据做任何解释
    texDesc.readMode = cudaReadModeElementType;
    //如果为1,表示使用归一化的坐标访问模式，也就是使用0~1的浮点数来表示自己想要访问的浮点数坐标
    //如果为0则使用数组被初始化的时候指定的数据的长和宽
    texDesc.normalizedCoords = 0;
    statu=cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if(statu!=cudaSuccess) std::cerr<<cudaGetErrorString(statu)<<std::endl;
	//把新建好的纹理内存句柄保存到dstObj里面
	*dstObj=texObj;
	//保存array的句柄
	*dstArray=cuArray;
}