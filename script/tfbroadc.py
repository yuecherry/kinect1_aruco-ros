#!/usr/bin/env python  
import roslib
import rospy

import tf

from aruco_msgs.msg import Marker
from aruco_msgs.msg import MarkerArray

def callback(msg):
    for marker in msg.markers:
        br = tf.TransformBroadcaster()
        br.sendTransform((marker.pose.pose.position.x,marker.pose.pose.position.y,marker.pose.pose.position.z),
                        (marker.pose.pose.orientation.x,marker.pose.pose.orientation.y,marker.pose.pose.orientation.z,marker.pose.pose.orientation.w),
                        rospy.Time.now(),
                        'arucomarker_' + str(marker.id),
                        marker.header.frame_id)

if __name__ == '__main__':
    rospy.init_node('aruco_tf_broadc')
    rospy.Subscriber('/aruco_marker_publisher_k1/markers', MarkerArray ,callback)
    def clean_shutdown():
        print("\nExiting...")
    rospy.on_shutdown(clean_shutdown)
    rospy.spin()