#!/usr/bin/env python

import vtk
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import vtk.util.numpy_support as vtknp
import time
from utils.util import get_slab_volume
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from visualization_msgs.msg import Marker,MarkerArray
import tf2_ros
import cv2
from scipy.io import savemat    
import time
# 全局变量
latest_pose = None
position_marker_pub = None
tf_broadcaster = None
previous_pose = None

# 将四元数转换为旋转矩阵
def quaternion_to_matrix(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def quaterion_mult(q_a, q_b):
    return np.array([
        q_a.w * q_b.w - q_a.x * q_b.x - q_a.y * q_b.y - q_a.z * q_b.z,
        q_a.w * q_b.x + q_a.x * q_b.w + q_a.y * q_b.z - q_a.z * q_b.y,
        q_a.w * q_b.y - q_a.x * q_b.z + q_a.y * q_b.w + q_a.z * q_b.x,
        q_a.w * q_b.z + q_a.x * q_b.y - q_a.y * q_b.x + q_a.z * q_b.w,
    ])

def quaterion_div(q_a, q_b):
    m_a = np.array([q_a.w, q_a.x, q_a.y, q_a.z])
    norm = np.sqrt(q_b.x**2 + q_b.y**2 + q_b.z**2 + q_b.w**2)
    result = m_a / norm
    return result

# 归一化四元数
def normalize_quaternion(q):
    norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    if norm == 0:
        rospy.logwarn("Quaternion norm is zero, returning original quaternion")
        return q
    return Quaternion(x=q.x/norm, y=q.y/norm, z=q.z/norm, w=q.w/norm)

# 位姿回调函数
def pose_callback(msg):
    global latest_pose, position_marker_pub, tf_broadcaster
    latest_pose = PoseStamped()
    p = msg.pose.position
    q = msg.pose.orientation

    p.x += 2*(q.x*q.z + q.w*q.y)*0.17421
    p.y += 2*(q.y*q.z - q.w*q.x)*0.17421
    p.z += (1 - 2*q.x**2 - 2*q.y**2)*0.17421
    latest_pose.pose.position = p
    latest_pose.pose.orientation = q
    
    marker_array = MarkerArray()
    position_marker = Marker()
    position_marker.header.frame_id = msg.header.frame_id
    position_marker.header.stamp = rospy.Time.now()
    position_marker.ns = "probe_position"
    position_marker.id = 0
    position_marker.type = Marker.SPHERE
    position_marker.action = Marker.ADD
    position_marker.pose.position = p
    position_marker.pose.orientation = Quaternion(0, 0, 0, 1)
    position_marker.scale.x = 0.05
    position_marker.scale.y = 0.05
    position_marker.scale.z = 0.05
    position_marker.color.r = 0.0
    position_marker.color.g = 1.0
    position_marker.color.b = 0.0
    position_marker.color.a = 1.0
    # position_marker_pub.publish(position_marker)
    
    text_marker = Marker()
    text_marker.header.frame_id = msg.header.frame_id  # 与 position_marker 相同
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "probe_position"
    text_marker.id = 1  # 不同 ID，避免冲突
    text_marker.type = Marker.TEXT_VIEW_FACING  # 文本类型
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = p.x
    text_marker.pose.position.y = p.y + 0.3
    text_marker.pose.position.z = p.z + 0.1  # 上移 0.1 米，避免重叠
    text_marker.pose.orientation = Quaternion(0, 0, 0, 1)
    text_marker.scale.z = 0.01  
    text_marker.color.r = 1.0
    text_marker.color.g = 0.0
    text_marker.color.b = 0.0
    text_marker.color.a = 1.0  
    text_marker.text = str(latest_pose.pose)  # 文本内容
    
    marker_array.markers.append(position_marker)
    marker_array.markers.append(text_marker)
    position_marker_pub.publish(marker_array)

    q_normalized = normalize_quaternion(q)
    R = quaternion_to_matrix(q_normalized)
    
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    x_transform = TransformStamped()
    x_transform.header.stamp = rospy.Time.now()
    x_transform.header.frame_id = msg.header.frame_id
    x_transform.child_frame_id = "probe_x_axis"
    x_transform.transform.translation.x = p.x + 0.1 * x_axis[0]
    x_transform.transform.translation.y = p.y + 0.1 * x_axis[1]
    x_transform.transform.translation.z = p.z + 0.1 * x_axis[2]
    x_transform.transform.rotation = Quaternion(0, 0, 0, 1)
    tf_broadcaster.sendTransform(x_transform)
    
    y_transform = TransformStamped()
    y_transform.header.stamp = rospy.Time.now()
    y_transform.header.frame_id = msg.header.frame_id
    y_transform.child_frame_id = "probe_y_axis"
    y_transform.transform.translation.x = p.x + 0.1 * y_axis[0]
    y_transform.transform.translation.y = p.y + 0.1 * y_axis[1]
    y_transform.transform.translation.z = p.z + 0.1 * y_axis[2]
    y_transform.transform.rotation = Quaternion(0, 0, 0, 1)
    tf_broadcaster.sendTransform(y_transform)
    
    z_transform = TransformStamped()
    z_transform.header.stamp = rospy.Time.now()
    z_transform.header.frame_id = msg.header.frame_id
    z_transform.child_frame_id = "probe_z_axis"
    z_transform.transform.translation.x = p.x + 0.1 * z_axis[0]
    z_transform.transform.translation.y = p.y + 0.1 * z_axis[1]
    z_transform.transform.translation.z = p.z + 0.1 * z_axis[2]
    z_transform.transform.rotation = Quaternion(0, 0, 0, 1)
    tf_broadcaster.sendTransform(z_transform)

# 定时器回调函数
def timer_callback(obj, event, reader, transform, _axes, spacingT2, T_vessel_to_base_inv, R_x, R_z, R_y, probe_actor, reslice, image_pub, slab_volume_pub, bridge, render_window, extentT2):
    global latest_pose, previous_pose
    if latest_pose is not None:
        start_time = time.time()
        delay = (rospy.Time.now() - latest_pose.header.stamp).to_sec()
        if delay > 0.1:
            pass
        
        p = latest_pose.pose.position
        q = latest_pose.pose.orientation
        # reslice.SetOutputOrigin(p.x, p.y, p.z)
        
        dx = dy = dz = 0.0
        q_real_norm = [0, 0, 0, 0]
        if previous_pose is not None:
            dx = p.x - previous_pose.pose.position.x
            dy = p.y - previous_pose.pose.position.y
            dz = p.z - previous_pose.pose.position.z
            
            previous_pose_conj = Quaternion(x=-previous_pose.pose.orientation.x, y=-previous_pose.pose.orientation.y, z=-previous_pose.pose.orientation.z, w=previous_pose.pose.orientation.w)
            q_real = quaterion_mult(previous_pose_conj, q)
            q_real = Quaternion(x=q_real[1], y=q_real[2], z=q_real[3], w=q_real[0])
            q_real_norm = quaterion_div(q_real, q_real).round(1)
        
        previous_pose = PoseStamped()
        previous_pose.header = latest_pose.header
        previous_pose.pose.position = Point(x=p.x, y=p.y, z=p.z)
        previous_pose.pose.orientation = q
        
        q_normalized = normalize_quaternion(q)
        R = quaternion_to_matrix(q_normalized)
        
        p_mm = [p.x * 1000, p.y * 1000, p.z * 1000]
        
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, R[i][j])
            matrix.SetElement(i, 3, p_mm[i])
        
        transform.Translate(0, 0, extentT2[5] * spacingT2[2])
        transform.Translate(dx, dy, dz)
        transform.RotateWXYZ(q_real_norm[0], q_real_norm[1], q_real_norm[2], q_real_norm[3])
        transform.Translate(0, 0, -extentT2[5] * spacingT2[2])
        
        probe_actor.SetUserMatrix(matrix)
        # axes_actor.SetUserMatrix(matrix)
        
        T_probe_to_vessel = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(T_vessel_to_base_inv, matrix, T_probe_to_vessel)
        
        T_probe_to_vessel_y = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(T_probe_to_vessel, R_y, T_probe_to_vessel_y)
        
        reslice.SetResliceAxes(T_probe_to_vessel_y)
        reslice.Update()
        
        output = reslice.GetOutput()
        
        image_array = vtknp.vtk_to_numpy(output.GetPointData().GetScalars())
        dim = output.GetDimensions()
        image_array = image_array.reshape(dim[1], dim[0])
        scalar_type = output.GetScalarType()
        if scalar_type == vtk.VTK_UNSIGNED_CHAR:
            encoding = "mono8"
        elif scalar_type == vtk.VTK_SHORT:
            encoding = "16UC1"
        elif scalar_type == vtk.VTK_FLOAT:
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-6) * 255
            image_array = image_array.astype(np.uint8)
            encoding = "mono8"
        else:
            rospy.logwarn("Unsupported scalar type")
            return
        cv2.imwrite("./original_us.png",image_array)
        print("######image########",np.shape(image_array))
        ros_image = bridge.cv2_to_imgmsg(image_array, encoding)
        image_pub.publish(ros_image)
        
        slab_volum = get_slab_volume(
            image_reader=reader,
            transform=transform,
            slab_thickness_mm=4,
            spacing=spacingT2[2],
            axes=T_probe_to_vessel_y
        )
        print("#####slab######",np.shape(slab_volum))
        
        n_slices, h, w = slab_volum.shape
        slab_msg = Float32MultiArray()
        slab_msg.layout.dim = [
            MultiArrayDimension(label="n_slices", size=n_slices, stride=w*h),
            MultiArrayDimension(label="h", size=h, stride=w),
            MultiArrayDimension(label="w", size=w, stride=1)
        ]
        slab_msg.data = slab_volum.flatten().tolist()
        slab_volume_pub.publish(slab_msg)
        
        render_window.Render()
        
        latest_pose = None
        
        exec_time = time.time() - start_time
        print("elastic time:", exec_time)

def main():
    
    global position_marker_pub, tf_broadcaster
    rospy.init_node('ultrasound_simulator')
    
    position_marker_pub = rospy.Publisher('/probe_position_marker', MarkerArray, queue_size=10)
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    
    rospy.Subscriber('/probe_pose', PoseStamped, pose_callback)
    
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName('/home/thera/expert_simulator_pa_7_5/src/ct_to_us_simulator/vessel.nii')
    reader.Update()
    spacingT2 = reader.GetOutput().GetSpacing()
    extentT2 = reader.GetOutput().GetExtent()
    
    transform = vtk.vtkTransform()
    transform.Translate(0, 0, 0)
    
    _axes = vtk.vtkMatrix4x4()
    _axes.SetElement(0, 2, 1)
    _axes.SetElement(1, 1, 1)
    _axes.SetElement(2, 0, -1)
    
    data = reader.GetOutput()
    scalar_range = data.GetScalarRange()
    bounds = data.GetBounds()
    center_vessel = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    
    p_desired = [0.31414382092127224 * 1000, 0.20010206254387328 * 1000, 0.07398700643920156 * 1000]
    q_desired = Quaternion(x=0.001554556123541251, y=0.9971732461084396, z=-0.07503339723093785, w=0.0036179967938902266)
    q_desired_normalized = normalize_quaternion(q_desired)
    R_desired = quaternion_to_matrix(q_desired_normalized)
    
    # 绕Y轴-90度的旋转矩阵
    R_y_neg90_np = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R_desired_new = np.dot(R_desired, R_y_neg90_np)
    
    t = np.array(p_desired) - np.dot(R_desired_new, center_vessel)
    
    T_vessel_to_base = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            T_vessel_to_base.SetElement(i, j, R_desired_new[i][j])
        T_vessel_to_base.SetElement(i, 3, t[i])
    
    T_vessel_to_base_inv = vtk.vtkMatrix4x4()
    T_vessel_to_base_inv.DeepCopy(T_vessel_to_base)
    T_vessel_to_base_inv.Invert()
    
    R_x = vtk.vtkMatrix4x4()
    R_x.SetElement(0, 0, 1)
    R_x.SetElement(0, 1, 0)
    R_x.SetElement(0, 2, 0)
    R_x.SetElement(0, 3, 0)
    R_x.SetElement(1, 0, 0)
    R_x.SetElement(1, 1, 0)
    R_x.SetElement(1, 2, -1)
    R_x.SetElement(1, 3, 0)
    R_x.SetElement(2, 0, 0)
    R_x.SetElement(2, 1, 1)
    R_x.SetElement(2, 2, 0)
    R_x.SetElement(2, 3, 0)
    R_x.SetElement(3, 0, 0)
    R_x.SetElement(3, 1, 0)
    R_x.SetElement(3, 2, 0)
    R_x.SetElement(3, 3, 1)
    
    R_y = vtk.vtkMatrix4x4()
    R_y.SetElement(0, 0, 0)
    R_y.SetElement(0, 2, 1)
    R_y.SetElement(1, 1, 1)
    R_y.SetElement(2, 0, -1)
    R_y.SetElement(0, 1, 0)
    R_y.SetElement(1, 0, 0)
    R_y.SetElement(1, 2, 0)
    R_y.SetElement(2, 1, 0)
    R_y.SetElement(2, 2, 0)
    
    R_z = vtk.vtkMatrix4x4()
    R_z.SetElement(0, 0, 0)
    R_z.SetElement(0, 1, -1)
    R_z.SetElement(0, 2, 0)
    R_z.SetElement(0, 3, 0)
    R_z.SetElement(1, 0, 1)
    R_z.SetElement(1, 1, 0)
    R_z.SetElement(1, 2, 0)
    R_z.SetElement(1, 3, 0)
    R_z.SetElement(2, 0, 0)
    R_z.SetElement(2, 1, 0)
    R_z.SetElement(2, 2, 1)
    R_z.SetElement(2, 3, 0)
    R_z.SetElement(3, 0, 0)
    R_z.SetElement(3, 1, 0)
    R_z.SetElement(3, 2, 0)
    R_z.SetElement(3, 3, 1)
    
    vessel_center_pub = rospy.Publisher('/vessel_center', PoseStamped, queue_size=1, latch=True)
    vessel_pose = PoseStamped()
    vessel_pose.header.frame_id = "base_link"
    vessel_pose.header.stamp = rospy.Time.now()
    vessel_pose.pose.position.x = p_desired[0]/1000
    vessel_pose.pose.position.y = p_desired[1]/1000
    vessel_pose.pose.position.z = p_desired[2]/1000
    vessel_pose.pose.orientation.x = q_desired.x
    vessel_pose.pose.orientation.y = q_desired.y
    vessel_pose.pose.orientation.z = q_desired.z
    vessel_pose.pose.orientation.w = q_desired.w
    vessel_center_pub.publish(vessel_pose)
    rospy.loginfo("Published vessel center pose to /vessel_center")
    
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    prop = vtk.vtkVolumeProperty()
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
    color.AddRGBPoint(scalar_range[0] + 0.25 * (scalar_range[1] - scalar_range[0]), 0.3, 0.3, 0.3)
    color.AddRGBPoint(scalar_range[0] + 0.5 * (scalar_range[1] - scalar_range[0]), 0.6, 0.6, 0.6)
    color.AddRGBPoint(scalar_range[0] + 0.75 * (scalar_range[1] - scalar_range[0]), 0.9, 0.9, 0.9)
    color.AddRGBPoint(scalar_range[1], 1.0, 1.0, 1.0)
    prop.SetColor(color)
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(scalar_range[0], 0.0)
    opacity.AddPoint(scalar_range[0] + 0.08 * (scalar_range[1] - scalar_range[0]), 0.5)
    opacity.AddPoint(scalar_range[1], 1.0)
    prop.SetScalarOpacity(opacity)
    prop.SetAmbient(0.3)
    volume.SetProperty(prop)
    volume.SetUserMatrix(T_vessel_to_base)
    
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName('/home/thera/expert_simulator_pa_7_5/src/linescan_mm.STL')
    stl_reader.Update()
    probe_mapper = vtk.vtkPolyDataMapper()
    probe_mapper.SetInputConnection(stl_reader.GetOutputPort())
    probe_actor = vtk.vtkActor()
    probe_actor.SetMapper(probe_mapper)
    probe_actor.RotateX(180)
    probe_actor.GetProperty().SetColor(1, 0, 0)
    
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.AddActor(probe_actor)
    renderer.SetBackground(0, 0, 0)
    
    # axes_actor = vtk.vtkAxesActor()
    # axes_actor.SetTotalLength(50, 50, 50)
    # renderer.AddActor(axes_actor)
    
    render_window = vtk.vtkOpenGLRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetMotionFactor(2)
    interactor.SetInteractorStyle(style)
    
    interactor.Initialize()
    
    renderer.ResetCamera()
    
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(reader.GetOutputPort())
    reslice.SetOutputDimensionality(2)
    reslice.SetOutputExtent(0, 255, 0, 255, 0, 0)
    reslice.SetOutputSpacing(0.2, 0.2, 1)
    
    image_pub = rospy.Publisher('/slice_image', Image, queue_size=10)
    slab_volume_pub = rospy.Publisher('/slab_volume', Float32MultiArray, queue_size=10)
    bridge = CvBridge()
    
    timer_interval_ms = 10
    
    interactor.CreateRepeatingTimer(timer_interval_ms)
    interactor.AddObserver('TimerEvent', lambda obj, event: timer_callback(obj, event, reader, transform, _axes, spacingT2, T_vessel_to_base_inv, R_x, R_z, R_y, probe_actor, reslice, image_pub, slab_volume_pub, bridge, render_window, extentT2))
    
    interactor.Start()
    

if __name__ == '__main__':
    main()