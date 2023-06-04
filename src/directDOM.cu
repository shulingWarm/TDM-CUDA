#include<iostream>
#include<string>
#include"readPoseCloud.hpp"
#include<vector>
#include"scene.hpp"
#include"grid.hpp"
#include"initGridByPointcloud.hpp"
#include"scenePropagate.hpp"
#include"exportSceneHeight.hpp"
#include"colorScene.hpp"
#include<thread>
#include"cpuColorCell.hpp"
#include"zmqServer.hpp"

//保存高度图的信息，需要先保存网格单元的大小，两个unsigned,然后保存3个float 分别是xymin和分辨率
void saveHeightMap(std::vector<float>& heightMap,Scene& scene,const std::string& heightFilePath)
{
	//准备输出流
	std::fstream fileHandle;
	fileHandle.open(heightFilePath,std::ios::out|std::ios::binary);
	if(!fileHandle.is_open())
	{
		std::cerr<<"cannot create file "<<heightFilePath<<std::endl;
		return;
	}
	//网格单元的相关信息
	auto& gridInfo=scene.grid->gridInfo;
	//需要确保网格数据的长度和scne里面的网格大小是一致的
	if(heightMap.size()!=gridInfo.gridSize[0]*gridInfo.gridSize[1])
	{
		std::cerr<<"height map length invalid"<<std::endl;
		fileHandle.close();
		return;
	}
	//记录网格单元的大小
	fileHandle.write((char*)(gridInfo.gridSize),sizeof(unsigned)*2);
	//记录xy的最小值
	fileHandle.write((char*)(gridInfo.xyMin),sizeof(float)*2);
	//记录空间分辨率
	fileHandle.write((char*)(&gridInfo.pixelResolution),sizeof(float));
	//记录所有的网格数据
	fileHandle.write((char*)(heightMap.data()),sizeof(float)*heightMap.size());
	fileHandle.close();
}

//载入所需需要的图片，由cpu单独开一个线程做这件事
void loadCvImage(std::vector<cv::Mat>* cpuImages,std::vector<ViewForSelect>* viewSelectList)
{
	//提前开辟图片的空间
	cpuImages->resize(viewSelectList->size());
	//遍历读取每个图片
	for(int idImage=0;idImage<viewSelectList->size();++idImage)
	{
		std::string imgPath=viewSelectList[0][idImage].imgPath;
		cpuImages[0][idImage]=cv::imread(imgPath);
	}
}

//试读取图片的长宽
void getImgSize(const std::string& imgPath,unsigned* imgSize)
{
	//读取图片
	cv::Mat tempImg=cv::imread(imgPath);
	//读取图片的大小
	imgSize[0]=tempImg.cols;
	imgSize[1]=tempImg.rows;
}

//把点云的最小值弄到0附近，并且还要排除范围之外的点云，然后还要返回一个平均高度，这里要处理很多事情
float pointHeightAdjust(std::vector<float>& pointcloud,
	std::vector<ViewForSelect>& viewList,float* range)
{
	//最小的z值
	float minZ=9999999;
	//z值的求和
	float sumZ=0;
	//把点复制一份
	std::vector<float> saveCloud;
	saveCloud.swap(pointcloud);
	//遍历所有的点
	for(int i=0;i+2<saveCloud.size();i+=3)
	{
		//判断是否在范围内
		if(isPointInRange(saveCloud.data()+i,range))
		{
			//记录这个点
			for(int j=0;j<3;++j)
			{
				pointcloud.push_back(saveCloud[i+j]);
			}
			//记录最小的z值
			if(saveCloud[i+2]<minZ)
			{
				minZ=saveCloud[i+2];
			}
			//z值求和结果的计数
			sumZ+=saveCloud[i+2];
		}
	}
	//判断有没有找到可用的点
	if(pointcloud.size()==0)
	{
		throw -1;
	}
	//计算平均z值
	float avgZ=sumZ*3/pointcloud.size();
	//z值平要完全减掉，留一点空间
	minZ+=1;
	//把平均值减掉最小值
	avgZ-=minZ;
	//给所有的z值减掉这个数
	for(int i=0;i+2<pointcloud.size();i+=3)
	{
		pointcloud[i+2]-=minZ;
	}
	//给所有的相机减掉最小的z值
	for(auto& eachView : viewList)
	{
		eachView.center[2]-=minZ;
	}
	//返回z的平均值
	return avgZ;
}

//把内参矩阵左乘所有的旋转矩阵
void adjustRotationMatByIntrinsic(float* intrinsic,
	std::vector<ViewForSelect>& viewList)
{
	//遍历每个图片
#pragma omp parallel for
	for(unsigned idImage=0;idImage<viewList.size();++idImage)
	{
		//当前位置的旋转
		auto* rotation= viewList[idImage].rotation;
		//开辟一个临时的结果
		float transAns[9];
		//计算每一行的结果
		for(int i=0;i<3;++i) transAns[i]=intrinsic[0]*rotation[i] + intrinsic[1]*rotation[6+i];
		for(int i=0;i<3;++i) transAns[3+i]=intrinsic[0]*rotation[3+i] + intrinsic[2]*rotation[6+i];
		for(int i=0;i<3;++i) transAns[6+i]=rotation[6+i];
		//把计算出来的结果只在回原始数据
		for(int i=0;i<9;++i) rotation[i]=transAns[i];
	}
}

//从stream里面载入生成正射影像需要的内容
void domGenerate(std::istream& taskHandle,cv::Mat& domResult)
{
	//读取网格单元的范围
	float range[4];
	taskHandle.read((char*)range,sizeof(float)*4);
	for(int i=0;i<4;++i)
	{
		std::cout<<range[i]<<" ";
	}
	std::cout<<std::endl;
	//读取空间分辨率
	float pixelResolution;
	taskHandle.read((char*)&pixelResolution,sizeof(float));
	std::cout<<pixelResolution<<std::endl;
	//接收相机内参
	float intrinsic[3];
	taskHandle.read((char*)intrinsic,sizeof(float)*3);
	//打印点云的信息
	// std::cout<<"intrinsic "<<std::endl;
	// for(int i=0;i<3;++i) std::cout<<intrinsic[i]<<" ";
	// std::cout<<std::endl;
	//收到的点云个数
	unsigned cloudNum;
	taskHandle.read((char*)&cloudNum,sizeof(unsigned));
	std::cout<<"cloudNum "<<cloudNum<<std::endl;
	//点云的平均高度
	float avgHeight;
	//接收里面的点云
	std::vector<float> pointcloud(cloudNum*3);
	taskHandle.read((char*)pointcloud.data(),sizeof(float)*pointcloud.size());
	//读取相机的个数
	unsigned poseNum;
	taskHandle.read((char*)&poseNum,sizeof(unsigned));
	std::cout<<"poseNum "<<poseNum<<std::endl;
	//开辟pose的个数
	std::vector<ViewForSelect> viewList(poseNum);
	//依次读取每个view的信息
	for(unsigned idPose=0;idPose<poseNum;++idPose)
	{
		//当前的目标view
		auto& targetView=viewList[idPose];
		//读取光心信息
		taskHandle.read((char*)targetView.center,sizeof(float)*3);
		//读取旋转信息
		taskHandle.read((char*)targetView.rotation,sizeof(float)*9);
		//读取路径的字符串长度
		unsigned strLength;
		taskHandle.read((char*)&strLength,sizeof(unsigned));
		//读取字符串
		targetView.imgPath.resize(strLength);
		taskHandle.read((char*)targetView.imgPath.data(),sizeof(char)*strLength);
	}
	//对所有的点做一个高度调整
	avgHeight=pointHeightAdjust(pointcloud,viewList,range);
	std::cout<<"avgHeight "<<avgHeight<<std::endl;
	//用内参矩阵左乘所有的旋转矩阵
	adjustRotationMatByIntrinsic(intrinsic,viewList);
	//一个示例图片的路径
	std::string exampleImgPath=viewList.front().imgPath;
	//试读取图片的长宽
	unsigned imgSize[2];
	getImgSize(exampleImgPath,imgSize);
	//初始化scene准备生成正射影像
	Scene cpuScene;
	cpuScene.init(range,pixelResolution,viewList,imgSize[0],imgSize[1]);
	//这里开一个线程记录所有需要被使用的图片，和cuda的贴图过程并行进行
	std::vector<cv::Mat> cpuImages;
	std::thread loadThread(loadCvImage,&cpuImages,&viewList);
	//把scene转换到cuda端
	Scene* cudaScene=cpuScene.toCuda();
	//用点云初始化网格 
	initSceneByPointcloud(pointcloud,cudaScene,&cpuScene);
	//进入scene传播的总流程 最后得到的是一个高程图的结果
	scenePropagate(&cpuScene,cudaScene,avgHeight);
	//根据cuda scene的高度，给scene染色
	colorScene(cudaScene,&cpuScene);
	//把cuda的scene复制到cpu端 主要是复制网格里面的内容
	cpuScene.fromCuda(cudaScene);
	//释放scene里面所有的纹理信息
	cpuScene.releaseTextures();
	//等待载入图片的线程完成运行
	loadThread.join();
	makeFinalDOM(cpuScene,cpuImages,domResult);
	//释放cpu的内容
	cpuScene.release();
}

int main(int argc,char** argv)
{
	//新建一个zmq连接
	ZmqConnection connection(5555,false);
	//死循环接收任务
	while(true)
	{
		std::stringstream tempStream;
		connection.receive(tempStream);
		//准备生成正射影像
		cv::Mat domResult;
		try
		{
			domGenerate(tempStream,domResult);
		}
		catch(...){}
		//发送图片的宽和高
		unsigned domSize[2]={domResult.cols,domResult.rows};
		connection.write((char*)domSize,sizeof(unsigned)*2);
		//写入图片每一行的数据，opencv行之间可能会为了对齐出现一些没用的空字节
		for(unsigned idRow=0;idRow<domResult.rows;++idRow)
		{
			connection.write((char*)domResult.ptr<uchar>(idRow),sizeof(uchar)*domResult.cols*3);
		}
		//把回传和消息送回去
		connection.send();
	}
}