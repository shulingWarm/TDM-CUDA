#pragma once
#include"config.hpp"
#include"grid.hpp"
#include"scene.hpp"

//染色用的网格单元
//虽然叫颜色网格单元，但是里面并没有颜色
//而是用来染色的相关信息
class ColorCell
{
public:
	float z;
	uint16_t idGroup;
};

//用于染色的scene，它的view载入的是彩色的图片
//它的网格单元和原始的网格单元有所不同，里面只有高度数据和所属的相机组数据
class SceneColor
{
public:
	//针对染色问题的网格单元列表
	Grid<ColorCell> grid;

	//所有的bgr颜色，尺寸和网格是一样的，单独列出来只是为了方便复制内存
	uchar* colorMap;

	//同时使用cpu和gpu的scene初始化用于染色的scene

};

