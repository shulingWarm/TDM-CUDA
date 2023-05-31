#pragma once

#include<iostream>
#include<vector>
#include<fstream>
#include<iostream>

//默认这里面的内参是乘过旋转的
void readPoseCloud(const std::string& cloudPose,
	std::vector<float>& pointcloud,std::vector<float>& camCenterList,std::vector<float>& camRotList,
	std::vector<unsigned>& poseIdList)
{
	//打开文件输入流
    std::fstream fileHandle;
    fileHandle.open(cloudPose,std::ios::in|std::ios::binary);
    //如果数据没打开需要报个错
    if(!fileHandle.is_open())
    {
        std::cerr<<"cannot open "<<cloudPose<<std::endl;
        return;
    }
    //读取点云的个数
    unsigned cloudNum;
    fileHandle.read((char*)&cloudNum,sizeof(unsigned));
    //读取pose的个数
    unsigned poseNum;
    fileHandle.read((char*)&poseNum,sizeof(unsigned));
    //遍历读取所有的点云
    pointcloud.reserve(cloudNum*3);
    for(unsigned idCloud=0;idCloud<cloudNum;++idCloud)
    {
    	//临时的三维点
    	double tempPoint[3];
    	fileHandle.read((char*)tempPoint,sizeof(double)*3);
    	for(int i=0;i<3;++i) pointcloud.push_back(tempPoint[i]);
    }
	//记录光心数据
    camCenterList.reserve(3*poseNum);
    camRotList.reserve(9*poseNum);
    //依次遍历所有的数据，读取pose
    for(unsigned idPose=0;idPose<poseNum;++idPose)
    {
        //临时的光心数据
        double camData[9];
        fileHandle.read((char*)camData,sizeof(double)*3);
        //记录光心数据
        for(int i=0;i<3;++i) camCenterList.push_back(camData[i]);
        //读取旋转的数据
        fileHandle.read((char*)camData,sizeof(double)*9);
        for(int i=0;i<9;++i)
        {
            camRotList.push_back(camData[i]);
            //std::cout<<camData[i]<<" ";
        }
        //std::cout<<std::endl;
    }
	//记录所有光心的标号
	poseIdList.resize(poseNum);
	fileHandle.read((char*)poseIdList.data(),sizeof(unsigned)*poseNum);
}