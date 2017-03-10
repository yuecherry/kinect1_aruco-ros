# kinect1_arucoROS
An example on how to estimate the pose of an aruco marker with kinect1 and ROS

# Instructions
* Launch a ROS kinect1 rgb image grabber
* Change the images topic and parameters names in launch/arucopublisher.launch according to your image grabber topics names:
 * <remap from="/camera_info" to="/kinect1/rgb/camera_info" />
 * <remap from="/image" to="/kinect1/rgb/image" />
 * <param name="camera_frame" value="/kinect1_rgb_optical_frame"/>

* Copy the launch file in a ros package
* Launch arucopublisher.launch with roslaunch  
* Run the python script arucopose.py that subscribes to /markers topic published by marker_publisher node and average 10 poses in SE3.  

# Notes
* The script arucopose.py contains hard-coded markerid (105)
* The launch file arucopublisher.launch contains default marker size value: <arg name="markerSize" default="0.181"/>
* By default the script writes two config files in ~/.ros directory:
 * calib_k1.json: contains the transformation matrix from the marker to the camera in json format
 * calib_k1q.json: contains the translation + quaternion(qx,qy,qz,qw) from the marker to the camera in json format