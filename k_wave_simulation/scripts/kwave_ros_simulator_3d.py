#!/home/thera/anaconda3/envs/kwave/bin python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import resize
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.kwave_array import kWaveArray
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG,kspaceFirstOrder3D,kspaceFirstOrder3DC
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import std_msgs.msg
from math import *

class KWaveROSNode:
    def __init__(self):
        # Initialize ROS node
        # rospy.init_node('kwave_simulator', anonymous=True)
        self.rate = rospy.Rate(0.1)
        while True:
        
        # Initialize k-Wave parameters
            dx = 0.2e-3  # [m]
            Nx, Ny, Nz = 256, 256, 21
            self.kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])

            # Define medium
            self.medium = kWaveMedium(
                sound_speed=1540,  # [m/s]
                density=1000,      # [kg/m^3]
                alpha_coeff=0.75,  # [dB/(MHz^y cm)]
                alpha_power=1.5
            )
            self.kgrid.makeTime(self.medium.sound_speed)

            # Initialize source
            self.source = kSource()
            self.source.p0 = np.zeros((Nx, Ny, Nz))
            self.source.p = None
            self.source.p_mask = np.zeros((Nx, Ny, Nz))  # Zero mask to bypass validation

            # Initialize kWaveArray
            self.karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
            element_num = 128
            element_pitch = 0.3e-3  # [m]
            element_width = 0.273e-3  # [m]
            element_length = 4e-3  # [m]

            # Add rectangular elements
            for ind in range(element_num):
                x_pos = -(element_num * element_pitch / 2 - element_pitch / 2) + ind * element_pitch
                self.karray.add_rect_element([0, x_pos, 0], element_length, element_width, [0, 90, 0])

            self.karray.set_array_position([-0.01, 0, 0], [0, 0, 0])

            # Generate sensor mask
            sensor_mask = self.karray.get_array_binary_mask(self.kgrid)
            self.sensor = kSensor(mask=sensor_mask)
            self.sensor.record = ['p']
            ######plot
            global x,y,z
            x, y, z = np.nonzero(sensor_mask)
            # Simulation options
            self.simulation_options = SimulationOptions(
                save_to_disk=True,
                data_cast='single',
                pml_size=7,
                smooth_p0=False,
            )
            self.execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

            # Initialize CvBridge for image conversion
            self.bridge = CvBridge()

            # Initialize source_array
            self.source_array = None

            # ROS Publisher
            self.image_pub = rospy.Publisher('/slab_volume_image', Image, queue_size=10)
            print("###########receiving data#################")
            # ROS Subscriber
            self.sub = rospy.Subscriber('/slab_volume', Float32MultiArray, self.callback)
            self.rate.sleep()
    def callback(self, msg):
        try:
            # Extract dimensions from layout (assuming [z, x, y] order)
            
            dims = [dim.size for dim in msg.layout.dim]
            print("###########test##########",dims)
            if len(dims) != 3:
                rospy.logerr("Received Float32MultiArray with incorrect dimensions")
                return

            # Convert to numpy array and reshape
            # source_array = np.array(msg.data, dtype=np.float32).reshape(dims)
            # source_array = source_array[:,0:256,0:256]
            shape = tuple(dim.size for dim in msg.layout.dim)
            source_array = np.array(msg.data).reshape(shape)
            # print(source_array.size())
            source_array = np.transpose(source_array, (1, 2, 0)) / np.max(source_array)  # permute to [z, x, y]
            # rospy.loginfo(f"Received source_array with shape: {source_array.shape}, max value: {np.max(source_array)}")
            
            # Map source to grid
            center = [self.kgrid.Nx/2, self.kgrid.Ny/2, self.kgrid.Nz/2]
            x_start = max(1, int(round(center[0] - 64)))
            x_end = min(self.kgrid.Nx, x_start + 127)
            y_start = max(1, int(round(center[1] - 128)))
            y_end = min(self.kgrid.Ny, y_start + 255)
            z_start = max(1, int(round(center[2] - 128)))
            z_end = min(self.kgrid.Nz, z_start + 255)

            self.source.p0 = np.zeros((self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz))
            self.source.p0[x_start-1:x_end, y_start-1:y_end, z_start-1:z_end] = source_array[
                0:(x_end-x_start+1), 0:(y_end-y_start+1), 0:(z_end-z_start+1)]

            # Visualize source (optional, commented out to avoid blocking)
            xx, yy, zz = np.nonzero(self.source.p0)
            # print(f"Source.p0 non-zero indices shapes: xx={xx.shape}, yy={yy.shape}, zz={zz.shape}")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(self.kgrid.x_vec[x], self.kgrid.y_vec[y], self.kgrid.z_vec[z], s=1, c='r')
            # # ax.scatter(self.kgrid.x_vec[xx], self.kgrid.y_vec[yy],10, s=1, c='b')########一层
            # ax.scatter(self.kgrid.x_vec[xx], self.kgrid.y_vec[yy], self.kgrid.z_vec[zz], s=10, c='b')#####多层
            # ax.set_xlabel('x [m]')
            # ax.set_ylabel('y [m]')
            # # ax.set_zlabel('z [m]')
            # ax.set_title('Source and Detector Positions in 3D')
            # ax.set_box_aspect([1, 1, 1])

            ax.scatter(self.kgrid.x_vec[x], self.kgrid.y_vec[y], self.kgrid.z_vec[z], s=1, c='r')
            ax.scatter(self.kgrid.x_vec[xx], self.kgrid.y_vec[yy], self.kgrid.z_vec[zz], s=10, c='b')

            # 设置轴标签
            # ax.set_xlabel('x [m]')
            # ax.set_ylabel('y [m]')
            # ax.set_zlabel('z [m]')  # 恢复 z 轴标签，确保完整性
            ax.set_title('Source and Detector Positions in 3D')

            # 获取数据的范围
            x_range = (min(self.kgrid.x_vec.min(), self.kgrid.x_vec[xx].min()), 
                    max(self.kgrid.x_vec.max(), self.kgrid.x_vec[xx].max()))
            y_range = (min(self.kgrid.y_vec.min(), self.kgrid.y_vec[yy].min()), 
                    max(self.kgrid.y_vec.max(), self.kgrid.y_vec[yy].max()))
            z_range = (min(self.kgrid.z_vec.min(), self.kgrid.z_vec[zz].min()), 
                    max(self.kgrid.z_vec.max(), self.kgrid.z_vec[zz].max()))

            # 设置刻度间隔为 0.01
            x_ticks = np.arange(np.floor(x_range[0] / 0.01) * 0.01, 
                                np.ceil(x_range[1] / 0.01) * 0.01 + 0.01, 0.01)
            y_ticks = np.arange(np.floor(y_range[0] / 0.01) * 0.01, 
                                np.ceil(y_range[1] / 0.01) * 0.01 + 0.01, 0.01)
            z_ticks = np.arange(np.floor(z_range[0] / 0.01) * 0.01, 
                                np.ceil(z_range[1] / 0.01) * 0.01 + 0.01, 0.01)

            # 应用刻度
            
            # ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
            # ax.set_zticks(z_ticks)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.grid(False)  #
            plt.show()

            # rospy.loginfo(f"source.p0 shape: {self.source.p0.shape}, non-zero points: {np.sum(self.source.p0 != 0)}")
            # rospy.loginfo(f"source.p is None: {self.source.p is None}")
            # rospy.loginfo(f"source.p_mask shape: {self.source.p_mask.shape}, non-zero points: {np.sum(self.source.p_mask)}")

            # Run simulation
            output = kspaceFirstOrder3DG(self.kgrid, self.source, self.sensor, self.medium, 
                                       self.simulation_options, self.execution_options)

            # Combine sensor data
            print("##########combining data##############")
            combined_sensor_data = self.karray.combine_sensor_data(self.kgrid, output['p'].T)
            rospy.loginfo(f"combined_sensor_data shape: {combined_sensor_data.shape}")

            # Convert to image (normalize to 0-255 for 8-bit grayscale)
            combined_sensor_data = np.abs(combined_sensor_data)  # Ensure non-negative
            data_max = np.max(combined_sensor_data)
            if data_max > 0:
                combined_sensor_data = (combined_sensor_data / data_max * 255).astype(np.uint8)
            else:
                combined_sensor_data = combined_sensor_data.astype(np.uint8)

            # Publish as Image message
            try:
                image_msg = self.bridge.cv2_to_imgmsg(combined_sensor_data, encoding="mono8")
                image_msg.header.stamp = rospy.Time.now()
                print("################publish data###################")
                self.image_pub.publish(image_msg)
                rospy.loginfo("Published combined_sensor_data as Image")
            except Exception as e:
                rospy.logerr(f"Failed to publish image: {e}")

        except Exception as e:
            rospy.logerr(f"Error processing Float32MultiArray: {e}")

    # def run(self):
    #     rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('kwave_simulator', anonymous=True)
        KWaveROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass