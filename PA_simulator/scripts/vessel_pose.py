#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import vtk
import numpy as np
from math import *
class VesselModelPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('vessel_model_publisher', anonymous=True)

        # Publishers
        self.mesh_marker_pub = rospy.Publisher('/vessel_mesh_marker', Marker, queue_size=10)
        # self.pointcloud_pub = rospy.Publisher('/vessel_pointcloud', PointCloud2, queue_size=10)

        # Subscriber for /vessel_pose
        self.__pose = None
        rospy.Subscriber('/vessel_center', PoseStamped, self.pose_callback, queue_size=1)

        # STL file path
        self.stl_path = '/home/thera/expert_simulator_pa_7_5/src/ct_to_us_simulator/vessel_centered.stl'

        # Read STL file
        self.reader = vtk.vtkSTLReader()
        self.reader.SetFileName(self.stl_path)
        self.reader.Update()
        self.polydata = self.reader.GetOutput()
        if self.polydata is None or self.polydata.GetNumberOfPoints() == 0:
            rospy.logerr("Failed to read STL file: {}".format(self.stl_path))
            return
        # rospy.loginfo("Number of points in STL: {}".format(self.polydata.GetNumberOfPoints()))

        # Create Mesh Marker for RViz
        self.mesh_marker = Marker()
        self.mesh_marker.header.frame_id = 'base_link'
        self.mesh_marker.type = Marker.MESH_RESOURCE
        self.mesh_marker.action = Marker.ADD
        self.mesh_marker.mesh_resource = 'file://' + self.stl_path
        self.mesh_marker.scale.x = 1.0  # No scaling
        self.mesh_marker.scale.y = 1.0
        self.mesh_marker.scale.z = 1.0
        self.mesh_marker.color.r = 1.0
        self.mesh_marker.color.g = 0.0
        self.mesh_marker.color.b = 0.0
        self.mesh_marker.color.a = 1.0  # Fully opaque
        self.mesh_marker.id = 0

        # Set publishing rate
        self.rate = rospy.Rate(10)  # 10 Hz
    def rotate_quaternion(self, orig_x, orig_y, orig_z, orig_w, axis_x, axis_y, axis_z, angle):

        # 计算旋转轴的模长
        axis_norm = sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        if axis_norm < 1e-6:
            raise ValueError("Rotation axis cannot be zero")
        
        # 归一化旋转轴
        ux = axis_x / axis_norm
        uy = axis_y / axis_norm
        uz = axis_z / axis_norm
        
        # 计算旋转四元数
        half_angle = angle / 2.0
        cos_half = cos(half_angle)
        sin_half = sin(half_angle)
        rot_w = cos_half
        rot_x = ux * sin_half
        rot_y = uy * sin_half
        rot_z = uz * sin_half
        
        # 四元数乘法：q_new = q_rot * q_orig
        # 公式：(w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
        w = rot_w * orig_w - (rot_x * orig_x + rot_y * orig_y + rot_z * orig_z)
        x = rot_w * orig_x + rot_x * orig_w + (rot_y * orig_z - rot_z * orig_y)
        y = rot_w * orig_y + rot_y * orig_w + (rot_z * orig_x - rot_x * orig_z)
        z = rot_w * orig_z + rot_z * orig_w + (rot_x * orig_y - rot_y * orig_x)
        
        # 归一化结果四元数
        norm = sqrt(w**2 + x**2 + y**2 + z**2)
        if norm < 1e-6:
            return [1.0, 0.0, 0.0, 0.0]  # 返回单位四元数
        return [x / norm, y / norm, z / norm, w / norm]

    def pose_callback(self, msg):
        """Callback for /vessel_pose, store received pose."""
        self.__pose = msg
        # rospy.loginfo("Received pose: x={}, y={}, z={}".format(
        #     msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))
    
    def vtk_to_pointcloud2(self, polydata, pose, frame_id):
        """Convert vtkPolyData to PointCloud2 with pose transformation."""
        points = []
        # Extract points
        for i in range(polydata.GetNumberOfPoints()):
            point = polydata.GetPoint(i)
            points.append(point)
        points = np.array(points)

        # Apply rotation from quaternion
        qw, qx, qy, qz = (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
        rotation = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        transformed_points = np.dot(points, rotation.T)  # Rotate points
        transformed_points += np.array([pose.position.x, pose.position.y, pose.position.z])  # Translate

        # Create PointCloud2
        header = rospy.Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.now()
        return pc2.create_cloud_xyz32(header, transformed_points.tolist())

    def run(self):
        """Main loop to publish Mesh Marker and PointCloud2."""
        while not rospy.is_shutdown():
            
            if self.__pose is not None:
                # Update Mesh Marker pose
                self.mesh_marker.header.stamp = rospy.Time.now()
                # self.__pose所有都除以1000
                self.mesh_marker.pose.position.x = self.__pose.pose.position.x
                self.mesh_marker.pose.position.y = self.__pose.pose.position.y
                self.mesh_marker.pose.position.z = self.__pose.pose.position.z-0.05
                ix = self.__pose.pose.orientation.x
                iy = self.__pose.pose.orientation.y
                iz = self.__pose.pose.orientation.z
                iw = self.__pose.pose.orientation.w
                # self.mesh_marker.pose.orientation.x = iw * 0.5 + ix * 0.5 + iy * 0.5 - iz * 0.5
                # self.mesh_marker.pose.orientation.y = iw * 0.5 - ix * 0.5 + iy * 0.5 + iz * 0.5
                # self.mesh_marker.pose.orientation.z = iw * 0.5 + ix * 0.5 - iy * 0.5 + iz * 0.5
                # self.mesh_marker.pose.orientation.w = iw * 0.5 - ix * 0.5 - iy * 0.5 - iz * 0.5
                # self.mesh_marker.pose = self.__pose.pose
                # self.mesh_marker.pose.position = self.__pose.pose.position
                
                quaternion = self.rotate_quaternion(ix, iy, iz, iw, 0, 0, 1, -pi /2)
                # quaternion = self.rotate_quaternion(quaternion_i[0],quaternion_i[1],quaternion_i[2],quaternion_i[3],0,1,0,pi / 2)
                self.mesh_marker.pose.orientation.x = quaternion[0]
                self.mesh_marker.pose.orientation.y = quaternion[1]
                self.mesh_marker.pose.orientation.z = quaternion[2]
                self.mesh_marker.pose.orientation.w = quaternion[3]
                self.mesh_marker_pub.publish(self.mesh_marker)

                # Generate and publish PointCloud2
                # pointcloud_msg = self.vtk_to_pointcloud2(self.polydata, self.__pose.pose, 'base_link')
                # self.pointcloud_pub.publish(pointcloud_msg)
            else:
                rospy.logwarn_once("Waiting for /vessel_pose message...")

            self.rate.sleep()

def main():
    try:
        publisher = VesselModelPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()