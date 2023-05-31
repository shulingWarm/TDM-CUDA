#pragma once
#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include<string>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<string>
#include<unistd.h>

//获得当前宿主机的ip地址
std::string getIpV4()
{
    struct ifaddrs *ifaddr, *ifa;
    char host[NI_MAXHOST];

    // Get a list of network interfaces
    if (getifaddrs(&ifaddr) == -1) {
        std::cerr << "getifaddrs() failed." << std::endl;
        return "";
    }

    // Traverse the list of network interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        // Check if the interface is an IPv4 interface
        if (ifa->ifa_addr->sa_family == AF_INET) {
            // Convert the interface address to a string
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) != 0) {
                std::cerr << "getnameinfo() failed." << std::endl;
                continue;
            }

            // Print the interface name and IPv4 address
            //std::cout << ifa->ifa_name << ": " << host << std::endl;
        }
    }

    // Free the memory allocated by getifaddrs()
    freeifaddrs(ifaddr);

    return std::string(host);
}

//连接句柄，把它弄成输入流的形式，方便使用一些
class ConnectHandle
{
    //记录连接句柄
    int connectFd;
public:
    //初始化的时候需要用一个监听得到的句柄来处理
    ConnectHandle(int connectFd)
    {
        this->connectFd=connectFd;
    }

    //读取信息 读取失败的情况下会返回false
    bool read(char* data,unsigned byteSize)
    {
        while(true)
        {
            //从连接的文件描述符里面读取数据
            int receiveFlag=recv(connectFd,data,byteSize,
                                 0 //这个参数一般写0,不知道做什么用的
            );
            //如果没有接收到消息就再等等
            if(receiveFlag<0)
            {
                continue;
            }
            else if(receiveFlag==0)
            {
                return false;
            }
            break;
        }
        return true;
    }

    //写入信息
    void write(char* data,unsigned byteSize)
    {
        if (send(connectFd, data,byteSize, 0) == -1) {
            std::cerr << "Error sending message" << std::endl;
            throw -1;
        }
    }

    //断开连接
    void disconnect()
    {
        close(connectFd);
    }
};

//监听器，指定端口号之后可以寻找监听socket
class Listener
{
    int idPort;
    std::string ipAddress;
    //用于监听的socket
    int listenSocket=-1;
    //服务器自己的socket
    sockaddr_in serverSock;
public:
    Listener(int idPort)
    {
        this->idPort=idPort;
        //记录ip地址
        ipAddress=getIpV4();
        //新建一个socket
        listenSocket=socket(AF_INET,SOCK_STREAM,0);
        if(listenSocket<0)
        {
            std::cerr<<"cannot create new socket"<<std::endl;
            throw -1;
        }
        //设置socket的协议族 使用IPV4协议族
        serverSock.sin_family=AF_INET;
        //设置服务器的IP地址
        serverSock.sin_addr.s_addr=inet_addr(ipAddress.c_str());
        //设置服务器的端口号，服务器和客户端需要使用同一个端口号
        //htons似乎是把端口号用转换成字节的形式
        serverSock.sin_port=htons(idPort);
        //把sockek文件描述符和套接字绑定
        int tempFlag=bind(listenSocket,(sockaddr*)&serverSock,sizeof(serverSock));
        //如果得到的数字小于0说明绑定失败了
        if(tempFlag<0)
        {
            std::cerr<<"bind failed"<<std::endl;
            throw tempFlag;
        }
        //设置服务器允许被客户端连接，这样的操作已经可以对serverSock_生效了
        tempFlag=listen(listenSocket,10);//这里的5表示监听缓冲区的长度
        //如果得到的数字小于0说明进入监听状态失败了
        if(tempFlag<0)
        {
            std::cerr<<"listen failed"<<std::endl;
            throw tempFlag;
        }
    }

    //通过监听获得连接句柄
    ConnectHandle getConnect()
    {
        //写死循环等待和新的客户端的连接
        while(true)
        {
            sockaddr_in cilentSocket;
            //因为后面需要传入一个地址，所以这里新建了一个变量
            socklen_t sockLength=sizeof(cilentSocket);
            //监听新的连接，观察是否有新的连接
            int newSocketFd=accept(listenSocket,(sockaddr*)&cilentSocket,
                                   &sockLength //地址的长度后来并没有被用来做什么，只是因为从博客抄了下来
            );
            //判断刚建立的连接是否有效
            if(newSocketFd>=0)
            {
                //到这里就算是找到了想要的文件描述符
                return ConnectHandle(newSocketFd);
            }
            //没找到的话就休息一下
            sleep(1);
        }
    }
};

//连接器
class Caller
{
    int idPort;
    int sock;
    sockaddr_in addr;
public:
    Caller(int idPort)
    {
        this->idPort=idPort;
        // 创建套接字
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            std::cerr << "Error creating socket" << std::endl;
        }

        //本地的ip地址
        std::string ipAddress=getIpV4();

        // 准备地址
        addr.sin_family = AF_INET;
        addr.sin_port = htons(idPort); // 端口号
        addr.sin_addr.s_addr = inet_addr(ipAddress.c_str()); // 本地IP

        // 连接到发送进程
        if (connect(sock, (sockaddr*)&addr, sizeof(addr)) == -1) {
            std::cerr << "Error connecting to server" << std::endl;
        }
    }

    ConnectHandle getConnect()
    {
        return ConnectHandle(sock);
    }
};
