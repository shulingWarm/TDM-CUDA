#pragma once
#include"gridInfo.hpp"
#include"grid.hpp"
#include<vector>
#include"domCell.hpp"
#include"view.hpp"
#include<string>
#include"loadImgTexture.hpp"
#include"cudaHandleError.hpp"

//初始化viewList
void initViewList(ViewInfo* viewList,unsigned viewNum,
	std::vector<float>& camCenter,std::vector<float>& camRot,
	std::vector<unsigned>& poseIdList,
	const std::string& imgRootPath)
{
	//遍历每个view数据，这里是确保所有的数据都是按照相同的顺序传入的
	for(unsigned idView=0;idView<viewNum;++idView)
	{
		//当前需要处理的view
		auto& thisView=viewList[idView];
		//复制光心
		for(int i=0;i<3;++i) thisView.center[i]=camCenter[idView*3 + i];
		//复制相机的旋转
		for(int i=0;i<9;++i) thisView.rotation[i]=camRot[idView*9 + i];
		//图片的标号
		unsigned thisImgId=poseIdList[idView];
		//图片的路径
		std::string imgPath=imgRootPath+"/"+std::to_string(thisImgId)+IMG_SUFFIX;
		//把图片读取进纹理内存
		loadImgObjArray(imgPath,&thisView.texObj,&thisView.cuArray);
	}
}

//临时用于选择范围的view信息
//这个类不能再随便扩充成员变量，成员变量的内容也不能变，因为它现在已经是两个进程之间的通信标准了
class ViewForSelect
{
public:
	float center[3];
	float rotation[9];
	//这个view的图片路径
	std::string imgPath;

	//构造函数，需要传入外部的数据，然后复制进来
	ViewForSelect(float* center,float* rotation,std::string imgPath)
	{
		for(int i=0;i<3;++i) this->center[i]=center[i];
		for(int i=0;i<9;++i) this->rotation[i]=rotation[i];
		this->imgPath=imgPath;
	}

	ViewForSelect(){}
};

//判断一个二维点是否在一个二维的范围内
char isPointInRange(float* point2d,float* range)
{
	return point2d[0]>range[0] && point2d[0]<range[1] &&
		point2d[1]>range[2] && point2d[1]<range[3];
}

//选择viewSelectList里面在范围内的相机，其它的相机直接就删了
void chooseViewsByRange(std::vector<ViewForSelect>& viewSelectList,float* domRange)
{
	//外扩的范围大小
	const float SPAN_RANGE = 100;
	//初始化一个扩展之后的range
	std::vector<float> expandRange={domRange[0]- SPAN_RANGE,
		domRange[1] + SPAN_RANGE, domRange[2]- SPAN_RANGE , domRange[3] + SPAN_RANGE};
	//把view的列表复制一份
	std::vector<ViewForSelect> viewListCopy;
	viewListCopy.swap(viewSelectList);
	//遍历所有复制过的列表
	for(auto& eachView : viewListCopy)
	{
		//判断光心是否在想要的范围内
		if(isPointInRange(eachView.center,expandRange.data()))
		{
			viewSelectList.push_back(eachView);
		}
	}
}

//使用已经筛选过的相机来载入相机模型
void loadViewsByChooseAns(std::vector<ViewForSelect>& viewSelectList,ViewInfo* viewList)
{
	//遍历载入每个相机的信息
	for(unsigned idView=0;idView<viewSelectList.size();++idView)
	{
		//当前需要处理的view
		auto& thisView=viewList[idView];
		//复制光心
		for(int i=0;i<3;++i) thisView.center[i]=viewSelectList[idView].center[i];
		//复制相机的旋转
		for(int i=0;i<9;++i) thisView.rotation[i]=viewSelectList[idView].rotation[i];
		//图片的路径
		std::string imgPath=viewSelectList[idView].imgPath;
		//把图片读取进纹理内存
		loadImgObjArray(imgPath,&thisView.texObj,&thisView.cuArray);
	}
}

//这里是描述一个场景需要用到的所有信息，主要是为了方便信息传递，但耦合性会增加
class Scene
{
public:

	typedef Grid<DOMCell> GridType;

	//存储主要信息的网格单元
	GridType* grid=nullptr;

	//view信息的列表
	ViewInfo* viewList=nullptr;
	//view的个数
	unsigned viewNum=0;

	//每个图片的基本信息
	unsigned imgSize[2];

	//把里面的数据转换成cuda
	Scene* toCuda()
	{
		//新建一个空的对象，这个对象会浅拷贝当前的对象
		//如果对象里面有指针成员的话，则需要另外先转换里面的成员变量
		Scene tempScene=*this;
		//把网格数据转换成cuda
		tempScene.grid=this->grid->toCuda();
		//把view信息转换到cuda
		cudaMalloc((void**)(&tempScene.viewList),sizeof(ViewInfo)*viewNum);
		cudaMemcpy(tempScene.viewList,this->viewList,sizeof(ViewInfo)*viewNum,cudaMemcpyHostToDevice);
		//新建一个空的指针
		Scene* cudaScene;
		//直接开辟就可以，目前scene里面没有需要复制的成员
		cudaMalloc((void**)&cudaScene,sizeof(Scene));
		//把内容复制到scene里面
		cudaMemcpy(cudaScene,&tempScene,sizeof(Scene),cudaMemcpyHostToDevice);
		return cudaScene;
	}

	//从cuda端复制数据到类内
	//需要特别注意的是，这个函数只是用来回收计算结果的，所以和计算结果无关的信息比如view的信息就不再计算了
	//注意，用完了直接释放 毕竟释放的时候走的也是这个流程 但后期如果需要的话可以加一个功能来控制这个顺便释放的问题
	void fromCuda(Scene* cudaScene)
	{
		//把它简单先复制到cpu端
		Scene tempScene;
		handleError(cudaMemcpy(&tempScene,cudaScene,sizeof(Scene),cudaMemcpyDeviceToHost));
		//拿出来cpu端的网格，复制到当前的类内网格
		grid->fromCuda(tempScene.grid);
		//释放scene的本体
		handleError(cudaFree(cudaScene));
	}

	//释放scene里面所有的纹理信息
	void releaseTextures()
	{
		//遍历每个view
		for(unsigned idView=0;idView<viewNum;++idView)
		{
			//当前位置的view
			auto& thisView=viewList[idView];
			handleError(cudaDestroyTextureObject(thisView.texObj));
			handleError(cudaFreeArray(thisView.cuArray));
		}
	}

	//使用view的列表生成的正射影像
	//到最后会直接把选下来的图片放进viewList里面
	void init(float* domRange,float pixelLegnth,std::vector<ViewForSelect>& viewList,
		unsigned imgWidth,unsigned imgHeight)
	{
		//初始化一个gridInfo,用于初始化网格信息
		GridInfo gridInfo(domRange,pixelLegnth);
		//开辟网格单元的空间
		grid=(GridType*)malloc(sizeof(GridType));
		//用网格单元的大小初始化网格单元 需要先准备每个网格单元里面的信息
		grid->init(gridInfo);
		//初始化被选中的view列表，这里需要根据view的列表做一下初步筛选
		chooseViewsByRange(viewList,domRange);
		//开辟view的个数
		viewNum=viewList.size();
		this->viewList=(ViewInfo*)malloc(sizeof(ViewInfo)*viewNum);
		//初始化所有的相机信息，主要是pose和图片的纹理，但是它原始对应的图片标号就不需要存下来了
		loadViewsByChooseAns(viewList,this->viewList);
		//记录图片的长宽
		imgSize[0]=imgWidth;
		imgSize[1]=imgHeight;
	}

	//释放空间，不写析构函数是为了以后在其它地方使用
	void release()
	{
		if(viewList)
		{
			//释放view的列表
			free(viewList);
			viewList=nullptr;
		}
		//释放网格单元
		if(grid)
		{
			//释放grid的空间
			grid->release();
			//释放grid
			free(grid);
			grid=nullptr;
		}
	}
};