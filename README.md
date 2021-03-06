# kinect1_arucoROS
An example on how to estimate the pose of an aruco marker with kinect1 and ROS

# Instructions
* Launch a ROS kinect1 rgb image grabber
* Change the images topic and parameters names in launch/arucopublisher.launch according to your image grabber topics names:
	* ```xml<remap from="/camera_info" to="/kinect1/rgb/camera_info" />```
 	* ```xml<remap from="/image" to="/kinect1/rgb/image" />```
 	* ```xml<param name="camera_frame" value="/kinect1_rgb_optical_frame"/>```
	* ```xml<arg name="markerSize" default="0.181"/>```

* Copy the package in a ros workspace
* Launch arucopublisher.launch with roslaunch
* Use rqt_image_view to check if aruco marker is tracked (/result/image topic)
* Run the python script arucopose.py that subscribes to /markers topic published by marker_publisher node and average 10 poses in SE3.  

# Notes
* The script arucopose.py contains hard-coded markerid (105)
* By default the script writes two config files in ~/.ros directory:
 	* calib_k1.json: contains the transformation matrix from the marker to the camera in json format
 	* calib_k1q.json: contains the translation + quaternion(qx,qy,qz,qw) from the marker to the camera in json format
* The data in calib_k1q.json can be use with static_transform_publisher node of tf package to publish the transformation