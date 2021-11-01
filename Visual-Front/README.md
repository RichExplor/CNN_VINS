**SuperPoint_GhostNet Feature Detection** 
# 视觉前端特征提取，取代VINS的光流跟踪

# 参考
主要参考代码【pytorch-superpoint】 https://github.com/eric-yyjau/pytorch-superpoint.git
其中前端特征编码网络进行修改

### Requirements
* python 3.6
* torch 1.5.1
* torchvision 0.6.0
* rospy
* cv_bridge
* numpy 
* openCV 3.4.3
* 

### training Dataset
 [coco2014 (train2014, val2014)](https://cocodataset.org/#download)
 
### Usage
# 1、将预训练模型放到当前文件夹下，并运行脚本文件（第一种方式）
将 run_feature.py 中的虚拟环境名更换为自己的虚拟环境名，然后运行下面指令
bash run_feature.sh

# 2. 虚拟环境下运行（第二种方式）

source activate your_evironment
source ~/catkin_pytorch/install/setup.bash --extend
python feature_match_node.py 

# 3、根据需要，更改/添加参数文件的内容，包括模型文件路径、是否显示特征跟踪窗口等
