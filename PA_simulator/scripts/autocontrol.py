#!/usr/bin/env python

import rospy
import tf
import numpy as np
from trac_ik_python.trac_ik import IK
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion

joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

current_joints = [0.0] * 6
trajectory_points = []

def joint_states_callback(msg):
    global current_joints
    if len(msg.name) == 6:
        current_joints = list(msg.position)

def update_trajectory_marker(listener, marker_pub):
    global trajectory_points
    try:
        (trans, rot) = listener.lookupTransform('/base_link', '/wrist_3_link', rospy.Time(0))
        point = Point()
        q_x = rot[0]
        q_y = rot[1]
        q_z = rot[2]
        q_w = rot[3]
        point.x = trans[0] + 2*(q_x*q_z + q_w*q_y)*0.17421
        point.y = trans[1] + 2*(q_y*q_z - q_w*q_x)*0.17421
        point.z = trans[2] + (1 - 2*q_x**2 - 2*q_y**2)*0.17421
        
        trajectory_points.append(point)

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.points = trajectory_points
        marker_pub.publish(marker)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        pass

def generate_snake_path():
    x_min, x_max = 0.26, 0.36
    y_min, y_max = 0.06, 0.33
    z = 0.2  # 调整Z轴高度，避免不可达
    dy = 0.01
    dx = 0.001
    path = []
    direction = 1
    for y in np.arange(y_min, y_max + dy / 2, dy):
        if direction == 1:
            xs = np.arange(x_min, x_max + dx / 2, dx)
        else:
            xs = np.arange(x_max, x_min - dx / 2, -dx)
        for x in xs:
            path.append((x, y, z))
        direction *= -1
    return path

if __name__ == '__main__':
    rospy.init_node('arm_path_planning')

    # IK solver with orientation tolerance
    ik_solver = IK("base_link", "wrist_3_link")

    joint_cmd_pub = rospy.Publisher('/joint_commands', JointState, queue_size=10)
    marker_pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=10)
    rospy.Subscriber('/joint_states', JointState, joint_states_callback)

    listener = tf.TransformListener()

    rate = rospy.Rate(10)

    path = generate_snake_path()
    rospy.loginfo("Generated path with {} points".format(len(path)))

    target_orientation = Quaternion(-0.7029620949943417, 0.7111564837283072, 0.007424205949313796, 0.006754984706010551)
    last_solution = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # 中立种子
    
    for i, pose in enumerate(path):
        if rospy.is_shutdown():
            break

        x, y, z = pose
        qx, qy, qz, qw = target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w

        solution = ik_solver.get_ik(last_solution, x, y, z, qx, qy, qz, qw)
        if solution:
            last_solution = list(solution)
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.name = joint_names
            js.position = list(solution)
            joint_cmd_pub.publish(js)
            rospy.loginfo("Published joint command for pose: ({}, {}, {})".format(x, y, z))

            # 在第一个点和最后一个点停留60秒
            if i == 1:
                zero_pose = [x,y,z]
                zero_quan = [qx,qy,qz,qw]
            # if i == 10:
            #     rospy.loginfo("Pausing for 60 seconds at first point")
            #     rospy.sleep(30)
            
        else:
            rospy.logwarn("No IK solution for pose: ({}, {}, {}, {}, {}, {}, {})".format(x, y, z, qx, qy, qz, qw))

        update_trajectory_marker(listener, marker_pub)
        rate.sleep()
    zero_solution = ik_solver.get_ik(last_solution,zero_pose[0],zero_pose[1],zero_pose[2],zero_quan[0],zero_quan[1],zero_quan[2],zero_quan[3])
    if zero_solution:
            
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.name = joint_names
            js.position = list(zero_solution)
            joint_cmd_pub.publish(js)
    rospy.spin()