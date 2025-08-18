#!/usr/bin/env python3
# coding=utf-8

import rospy
import tf
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, InteractiveMarkerFeedback
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from trac_ik_python.trac_ik import IK
import math

class InteractiveMarkerController:
    def __init__(self, base_link, tip_link):
        # 初始化发布者
        self.joint_pub = rospy.Publisher('/ur3_probe_joint_states', JointState, queue_size=10)
        self.probe_pub = rospy.Publisher('/probe_pose', PoseStamped, queue_size=10)
        
        # 定义UR3的关节名称和初始位置
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # self.current_positions = [0.0] * 6
        self.current_positions = [0.4032901411250173, -1.1035183303468257, 1.5130605523638065, -1.9874471535940663, -1.5520226214471324, 0.4149461350970618]
        self.joint_state = JointState()
        self.joint_state.name = self.joint_names
        self.joint_state.position = self.current_positions
        self.joint_state.velocity = [0.0] * 6
        self.joint_state.effort = [0.0] * 6
        
        # 初始化TRAC-IK求解器
        self.base_link = base_link
        self.tip_link = tip_link
        self.ik_solver = IK(base_link, tip_link)
        
        # 初始化交互式标记服务器和TF监听器
        self.server = InteractiveMarkerServer("ur3_interactive_markers")
        self.listener = tf.TransformListener()
        
        # 等待TF变换可用
        self.wait_for_tf()
        
        # 发布初始关节状态
        self.publish_joint_state()
        
        # 创建交互式标记
        self.create_marker()
        
        # 设置定时器以10Hz更新标记位姿和关节状态
        rospy.Timer(rospy.Duration(0.1), self.update_marker_pose)

    def wait_for_tf(self):
        rospy.loginfo("等待TF变换从 {} 到 {}...".format(self.base_link, self.tip_link))
        while not rospy.is_shutdown():
            try:
                self.listener.waitForTransform(self.base_link, self.tip_link, rospy.Time(0), rospy.Duration(1.0))
                trans, rot = self.listener.lookupTransform(self.base_link, self.tip_link, rospy.Time(0))
                rospy.loginfo("TF变换已可用。")
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("TF变换尚未可用，正在重试...")
                rospy.sleep(1.0)

    def create_marker(self):
        # 获取初始位姿
        try:
            trans, rot = self.listener.lookupTransform(self.base_link, self.tip_link, rospy.Time(0))
            initial_pose = Pose()
            initial_pose.position.x = trans[0]
            initial_pose.position.y = trans[1]
            initial_pose.position.z = trans[2]
            initial_pose.orientation = Quaternion(*rot)
            initial_pose.orientation = self.normalize_quaternion(initial_pose.orientation)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF变换失败: {e}")
            rospy.loginfo("请检查URDF、robot_state_publisher和末端执行器链接名称")
            rospy.signal_shutdown("TF初始化失败")
            return

        # 创建交互式标记
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_link
        int_marker.name = "probe"
        int_marker.description = "UR3末端执行器控制"
        int_marker.pose = initial_pose

        # 添加6自由度控制
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        int_marker.controls.append(control)

        # 添加可视化标记（球体）
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
        control.markers.append(marker)
        int_marker.controls.append(control)

        # # 添加偏航（绕Z轴旋转）控制
        # yaw_control = InteractiveMarkerControl()
        # yaw_control.orientation.w = 1
        # yaw_control.orientation.x = 0
        # yaw_control.orientation.y = 0
        # yaw_control.orientation.z = 1
        # yaw_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # yaw_control.name = 'yaw'
        # int_marker.controls.append(yaw_control)

        # # 添加俯仰（绕X轴旋转）控制
        # pitch_control = InteractiveMarkerControl()
        # pitch_control.orientation.w = 1
        # pitch_control.orientation.x = 1
        # pitch_control.orientation.y = 0
        # pitch_control.orientation.z = 0
        # pitch_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # pitch_control.name = 'pitch'
        # int_marker.controls.append(pitch_control)

        # # 添加滚转（绕Y轴旋转）控制
        # roll_control = InteractiveMarkerControl()
        # roll_control.orientation.w = 1
        # roll_control.orientation.x = 0
        # roll_control.orientation.y = 1
        # roll_control.orientation.z = 0
        # roll_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # roll_control.name = 'roll'
        # int_marker.controls.append(roll_control)

        

        self.server.insert(int_marker, self.process_feedback)
        self.server.applyChanges()

    def normalize_quaternion(self, q):
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if norm == 0:
            return q
        return Quaternion(x=q.x/norm, y=q.y/norm, z=q.z/norm, w=q.w/norm)

    def process_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            new_pose = feedback.pose
            new_pose.orientation = self.normalize_quaternion(new_pose.orientation)
            solution = self.ik_solver.get_ik(self.current_positions,
                                             new_pose.position.x, new_pose.position.y, new_pose.position.z,
                                             new_pose.orientation.x, new_pose.orientation.y,
                                             new_pose.orientation.z, new_pose.orientation.w)
            if solution:
                self.current_positions = list(solution)
                self.joint_state.position = self.current_positions
                self.publish_joint_state()
            else:
                rospy.logwarn("未找到给定位姿的IK解")

    def publish_joint_state(self):
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_pub.publish(self.joint_state)

    def update_marker_pose(self, event):
        try:
            trans, rot = self.listener.lookupTransform(self.base_link, self.tip_link, rospy.Time(0))
            probe_pose = PoseStamped()
            probe_pose.header.stamp = rospy.Time.now()
            probe_pose.header.frame_id = self.base_link
            probe_pose.pose.position.x = trans[0]
            probe_pose.pose.position.y = trans[1]
            probe_pose.pose.position.z = trans[2]
            probe_pose.pose.orientation = Quaternion(*rot)
            probe_pose.pose.orientation = self.normalize_quaternion(probe_pose.pose.orientation)
            
            self.probe_pub.publish(probe_pose)
            self.server.setPose("probe", probe_pose.pose)
            self.server.applyChanges()
            self.publish_joint_state()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"无法获取从{self.base_link}到{self.tip_link}的TF变换")

def main():
    rospy.init_node('custom_joint_state_publisher', anonymous=True)
    
    # 获取末端执行器链接名称（通过参数可配置）
    end_effector_link = rospy.get_param('~end_effector_link', 'wrist_3_link')
    
    # 创建InteractiveMarkerController实例
    imc = InteractiveMarkerController('base_link', end_effector_link)
    
    # 主循环保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
