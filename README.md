# CNN-VINS 环境搭建及运行


######################################################################################

## 1. python环境搭建（支持ROS与python3共存）

### 1.1 anaconda安装python虚拟环境
#### 第一步：下载anaconda，并更换下载源镜像，提升下载速度
具体博客可参考本人主页：【https://blog.csdn.net/qq_37568167/article/details/105620960】

流程如下：
##### 下载anaconda
Anacond下载地址: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
选择支持python3.6的版本，找一个linux版本的下载

##### 添加可执行命令
sudo chmod +x Anaconda3-5.2.0-Linux-x86_64.sh

##### 全程 yes +回车,中途会询问默认安装目录,默认安装在/home下
bash ./Anaconda3-5.2.0-Linux-x86_64.sh

##### 添加清华镜像源，运行完之后,重启一下电脑
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

#### 第二步：创建pytorch虚拟环境

##### 虚拟环境创建，并激活
conda create -n pytorch python=3.6

source activate pytorch

##### 安装所需的依赖库文件， Requirements
* python 3.6
* torch 1.5.0
* torchvision 0.6.0
* rospy
* cv_bridge
* numpy 
* openCV 3.4.1.15

需要安装pytorch的GPU版本，笔者安装的是CUDA=10.1, 注意：电脑事先需要安装好cuda加速驱动，具体安装方式可参考英伟达官网以及相关博客

依赖库文件安装方法如下：

conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1

其他库安装方法,采用pip安装，使用清华源，例如安装python-opencv

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-opencv==3.4.1.15

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xxx(功能包的名字)


#### 第三步： 编译cv_bridge功能包
注意：需要关闭上面创建的虚拟环境（或者新开一个终端即可）

具体可参考下面：
参考：【https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3】
参考：【https://blog.csdn.net/weixin_44060400/article/details/104347628】

##### 安装依赖
sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge
##### 创建ros工作空间
mkdir catkin_ws

cd catkin_ws

catkin init
##### 设置cmake编译时的参数
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so

catkin config --install
##### Clone cv_bridge src
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
##### Find version of cv_bridge in your repository
apt-cache show ros-kinetic-cv-bridge | grep Version

    Version: 1.12.8-0xenial-20180416-143935-0800
##### Checkout right version in git repo. In our case it is 1.12.8
cd src/vision_opencv/

git checkout 1.12.8

cd ../../
##### Build
catkin build cv_bridge
##### Extend environment with new package
source install/setup.bash --extend


######################################################################################
## 2. 运行前端特征提取

注意，整个过程时在虚拟环境下运行，如下所示：

### 2.1 虚拟环境下运行（第一种方式）

#### 启动虚拟环境
source activate your_evironment  

source ~/catkin_pytorch/install/setup.bash --extend

#### 运行程序
cd Visual-Front-new

python feature_match_node.py 

### 2.2 直接使用提供的.sh脚本运行
cd Visual-Front-new
#### 注意 将 run_feature.py 中的虚拟环境名更换为自己的虚拟环境名，然后运行下面指令
bash run_feature.sh

#### 2.3 根据需要，更改/添加参数文件的内容，包括模型文件路径、是否显示特征跟踪窗口等

根据需要自行更改的几个地方：

feature_match_node.py文件：CamearIntrinsicParam（相机内参）, 接受话题名

feature_process.py文件： SuperPointFrontend_torch类中 self.net，根据选择的模型更改（默认使用SuperPointNet_GhostNet）

parameter.py: 图像（H,W）, 模型名称， 放缩尺度， 是否显示特征跟踪窗口等

######################################################################################
## 3. 运行VINS-Mono后端
将VINS-Mono使用catkin编译，编译完成后开始运行

### 3.1 运行启动文件.launch
#### 3.1.1 EuRoc数据集
roslaunch vins_estimator superpoint.launch
#### 3.1.2 桃子湖数据集
roslaunch vins_estimator superpoint_mynteye.launch

### 3.2 运行rviz显示
roslaunch vins_estimator vins_rviz.launch

### 3.3 运行bag数据包
rosbag play MH_04_difficult.bag


######################################################################################
## 4. 运行总结

## 运行顺序，由上到下依次运行

cd Visual-Front-new

bash run_feature.sh

roslaunch vins_estimator superpoint.launch

roslaunch vins_estimator vins_rviz.launch

rosbag play MH_04_difficult.bag
