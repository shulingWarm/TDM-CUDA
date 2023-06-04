# 介绍

## 这是什么？

这是一个服务端程序，用于生成数字正射影像，以进程通信的形式和系统中的其它进程做交互。

## 它的输入是什么？

1. 无畸变的图片

2. 图片的内参数(焦距、主点坐标)

3. 每个图片的外参数(光心坐标，旋转矩阵)

4. 稀疏点云

5. 空间分辨率(每个像素对应的空间距离)

## 怎样获得输出结果?

输出结果会以进程通过的形式将正射影像发送给发起请求的进程。

## 这个项目有什么来历？

本项目是这篇论文中算法的CUDA版实现：[论文链接](https://www.mdpi.com/2072-4292/15/1/177)

本项目有一个功能类似的cpu版本的代码实现：[github项目链接]([GitHub - shulingWarm/OrthographicDenseMatching](https://github.com/shulingWarm/OrthographicDenseMatching))

# 构建

## 依赖的第三方库

### CUDA

算法是基于CUDA实现的，可以使用以下命令判断CUDA是否被正确安装，能正常看到版本信息即可。

```shell
nvcc -V
```

### OpenCV

会用到它进行图片读写。

```shell
sudo apt install libopencv-dev
```

### ZeroMQ

会用到它进行进程间通信,**注意zmq库需要使用源码编译**，只有源码编译的库才会提供ConfigZeroMq的库才能被find_package找到。

```shell
git clone https://github.com/zeromq/libzmq.git
cd libzmq
mkdir build
cd build
cmake ..
make
sudo make install
```

## 构建的命令

```shell
mkdir build
cd build
export CC=/usr/local/cuda/bin/gcc
export CXX=/usr/local/cuda/bin/g++
cmake ..
make
```

# 其它问题

## 进程通信的格式是什么?

#### 请求格式

4个float: 表示生成的正射影像的空间范围，顺序为x_min,x_max,y_min,y_max

1个float: 表示正射影像的空间分辨率

3个float: 依次表示相机的焦距和主点x,y坐标

1个uint32: 表示点云的数量$N_P$

$3N_P$个float: 表示点云的数据，顺序为x1,y1,z1,x2,y2,z2...,xn,yn,zn

1个uint32: 表示图片的数量$N_C$

对于每个图片数据依次是3个float用于表示光心坐标,9个float用于表示旋转矩阵,1个uint32用于表示图片的绝对路径的长度$L_P$,$L_P$个char用于表示图片的路径。

#### 应答格式

2个uint32: 表示正射影像图片的宽度$W$和高度$H$

$3\times H\times W$个uchar: 表示正射影像图片的颜色数据，通道顺序为bgr。

## 能否在Windows下使用？

项目中依赖的库在windows下也很容易配置，是可以在Windows下使用的。

## 是否有对应的用户界面？

这个项目是有用户界面的，用户界面的链接:[项目链接](https://github.com/shulingWarm/TDM-CUDA-GUI)
