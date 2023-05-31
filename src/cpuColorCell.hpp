#pragma once
#include"scene.hpp"

//使用cpu里面记录过的颜色信息染色
void makeFinalDOM(Scene& cpuScene,std::vector<cv::Mat>& cpuImages,
	cv::Mat& finalDOM)
{
	//网格单元的大小
	auto& gridInfo=cpuScene.grid->gridInfo;
	//根据cpu 场景的大小初始化DOM
	finalDOM.create(gridInfo.gridSize[1],gridInfo.gridSize[0],CV_8UC3);
	std::cout<<"final DOM size "<<finalDOM.rows<<" "<<finalDOM.cols<<std::endl;
	//遍历最终结果里面的每个像素
	#pragma omp parallel for
	for(unsigned idRow=0;idRow<finalDOM.rows;++idRow)
	{
		//当前位置的颜色向量
		uchar* rowColors=finalDOM.ptr<uchar>(idRow);
		for(unsigned idCol=0;idCol<finalDOM.cols;++idCol)
		{
			//当前位置的颜色
			uchar* dstColor=rowColors+ idCol*3;
			//目标位置的网格单元 不一样的是，DOM里面存储的row是倒着存的，因此需要在这里给颜色取一个镜像
			auto& dstCell=(cpuScene.grid->getDataAt(idCol,finalDOM.rows - idRow -1))[0];
			//当前位置需要访问的图片
			auto& dstTexture=cpuImages[dstCell.idGroup];
			//当前点应该访问的图片坐标
			uint16_t* projectLocal=(uint16_t*)(&(dstCell.score));
			//判断投影点的位置是否在图片的范围内
			if(projectLocal[0]<dstTexture.cols && projectLocal[1]<dstTexture.rows)
			{
				//获取投影目标位置的颜色
				cv::Vec3b imageColor=dstTexture.at<cv::Vec3b>(projectLocal[1],projectLocal[0]);
				//保存读取到的颜色向量
				for(int i=0;i<3;++i) dstColor[i]=imageColor[i];
			}
			else
			{
				//把目标颜色写成黑色
				dstColor[0]=0;
				dstColor[1]=0;
				dstColor[2]=0;
			}
		}
	}
}