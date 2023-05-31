#pragma once
#include<zmq.hpp>
#include<string>
#include<sstream>

//基于zeromq的消息形式，属于一种连接句柄
//目前只支持本地连接
class ZmqConnection
{
	//zmq的上下文
	zmq::context_t context;
	//请求用的socket
	zmq::socket_t socket;
	//待发送的缓冲区
	std::stringstream sendBuffer;
	//由于buffer里面不方便读取长度，只能另外设置一个数值来记录缓冲区的长度
	unsigned bufferSize=0;

	//根据标号把它弄成ip地址
	//有一个特殊的地方，请求端需要写成本地的ip但响应端只需要写成*:port
	static std::string makeIpString(unsigned port,bool reqFlag)
	{
		//请求端
		if(reqFlag)
		{
			return "tcp://localhost:"+std::to_string(port);
		}
		return "tcp://*:"+std::to_string(port);
	}

	//根据请求的情况获得的socket类型
	static zmq::socket_type getSocketType(bool reqFlag)
	{
		if(reqFlag) return zmq::socket_type::req;
		return zmq::socket_type::rep;
	}

public:
	ZmqConnection(unsigned port,bool reqFlag)
	{
		//初始化contect
		this->context=zmq::context_t(1);
		//初始化socket
		socket=zmq::socket_t(context,getSocketType(reqFlag));
		//如果是请求端就使用connect,否则使用bind
		if(reqFlag)
		{
			socket.connect(makeIpString(port,reqFlag));
		}
		else
		{
			socket.bind(makeIpString(port,reqFlag));
		}
	}

	//向buffer里面写入，由于请求信息只能一次性发送，所以只能使用这种形式，把要发送的全放进去再处理
	void write(const char* data,unsigned byteSize)
	{
		sendBuffer.write(data,byteSize);
		//记录buffer里面的长度
		bufferSize+=byteSize;
	}

	//发送缓冲区里的内容
	void send()
	{
        //如果buffer里面有东西再发送
		if(bufferSize==0) return;
		//新建相同大小的message
		zmq::message_t sendMessage(bufferSize);
		//从buffer里面复制数据
		sendBuffer.read((char*)sendMessage.data(),bufferSize);
		//把数据发出去
		socket.send(sendMessage,zmq::send_flags::none);
		//发出去后把数据清空
		sendBuffer.clear();
		bufferSize=0;
	}

	//接收响应数据
	void receive(std::stringstream& recvBuffer)
	{
		//接收响应用的消息
		zmq::message_t recvMessage;
		socket.recv(recvMessage,zmq::recv_flags::none);
		//把收到的数据复制到buffer里面
		recvBuffer.write((char*)recvMessage.data(),recvMessage.size());
	}

	//定长度复制 调用这个函数需要自己预判会收到多少数据，提前给data开辟好空间
	void receive(char* data)
	{
		//接收响应用的消息
		zmq::message_t recvMessage;
		socket.recv(recvMessage,zmq::recv_flags::none);
		//把收到的数据复制到buffer里面
		memcpy(data,recvMessage.data(),recvMessage.size());
	}
};
