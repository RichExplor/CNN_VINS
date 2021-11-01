import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import copy
import rospy
import torch

import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32


from time import time
from feature_match import VisualTracker
from utils.parameter import read_image, readParameters
from utils.camera_model import PinholeCamera

init_pub = False
count_frame = 0

def img_callback(img_msg, param):

    global init_pub
    global count_frame

    if not init_pub :
        init_pub = True
    else :
        init_pub = False

        bridge = CvBridge()
        conver_img = bridge.imgmsg_to_cv2(img_msg, "mono8")

        cur_img, status = read_image(conver_img, [param.height, param.width])

        if status is False:
            print("Load image error, Please check image_info topic")
            return

        param.readImage(cur_img)

        if True :

            feature_points = PointCloud()
            id_of_point = ChannelFloat32()
            u_of_point = ChannelFloat32()
            v_of_point = ChannelFloat32()
            velocity_x_of_point = ChannelFloat32()
            velocity_y_of_point = ChannelFloat32()
            feature_points.header = img_msg.header
            feature_points.header.frame_id = "world"

            cur_un_pts, cur_pts, ids = param.undistortedLineEndPoints( scale=param.scale )

            for j in range(len(ids)):
                un_pts = Point32()
                un_pts.x = cur_un_pts[0,j]
                un_pts.y = cur_un_pts[1,j]
                un_pts.z = 1

                feature_points.points.append(un_pts)
                id_of_point.values.append(ids[j])
                u_of_point.values.append(cur_pts[0,j])
                v_of_point.values.append(cur_pts[1,j])
                velocity_x_of_point.values.append(0.0)
                velocity_y_of_point.values.append(0.0)

            feature_points.channels.append(id_of_point)
            feature_points.channels.append(u_of_point)
            feature_points.channels.append(v_of_point)
            feature_points.channels.append(velocity_x_of_point)
            feature_points.channels.append(velocity_y_of_point)

            pub_img.publish(feature_points)

            ptr_toImageMsg = Image()

            ptr_toImageMsg.header = img_msg.header
            ptr_toImageMsg.height = param.height * param.scale
            ptr_toImageMsg.width = param.width * param.scale
            ptr_toImageMsg.encoding = 'bgr8'

            ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

            for pt in cur_pts.T:
                pt2 = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)

            ptr_toImageMsg.data = np.array(ptr_image).tostring()
            pub_match.publish(ptr_toImageMsg)


#####################################################################
###########################   main入口  ##############################
#####################################################################

if __name__ == '__main__':

  rospy.init_node('feature_tracker', anonymous=False)

  ############## 加载参数 #################
  Option_Param = readParameters()
  print(Option_Param)

  CamearIntrinsicParam = PinholeCamera(
      fx = 461.6, fy = 460.3, cx = 363.0, cy = 248.1, 
      k1 = -2.917e-01, k2 = 8.228e-02, p1 = 5.333e-05, p2 = -1.578e-04
      )  

#   CamearIntrinsicParam = PinholeCamera(
#       fx = 349.199951171875, fy = 349.199951171875, cx = 322.2005615234375, cy = 246.161865234375, 
#       k1 = -0.2870635986328125, k2 = 0.06902313232421875, p1 = 0.000362396240234375, p2 = 0.000701904296875
#       )
  FeatureParameters = VisualTracker(Option_Param, CamearIntrinsicParam)

#   sub_img = rospy.Subscriber("/mynteye/left/image_color", Image, img_callback, FeatureParameters,  queue_size=100)
  sub_img = rospy.Subscriber("/cam0/image_raw", Image, img_callback, FeatureParameters,  queue_size=100) 
 

  pub_img = rospy.Publisher("/feature_tracker/feature", PointCloud, queue_size=1000)
  pub_match = rospy.Publisher("/feature_tracker/feature_img", Image, queue_size=1000)

  rospy.spin()
