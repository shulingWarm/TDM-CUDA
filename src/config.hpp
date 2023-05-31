#pragma once
#include<string>

typedef unsigned short int uint16_t;
//分数的数据类型
typedef float ScoreType;
typedef unsigned char uint8_t;
typedef unsigned char uchar;

//分配每个网格单元属于哪个相机组的时候，分辨率不需要那么高，这个时候就需要有一个比例
const float GROUP_RESOLUTION_RATE = 8;
//具有已知高度的网格单元的初始分数
//ScoreType PRIOR_SCORE = 0.7*255;
#define PRIOR_SCORE 0.8
//每个相机组里面有几个相机
const unsigned IMG_GROUP_SIZE = 8;

//读取图片时的后缀
const std::string IMG_SUFFIX = ".JPG";