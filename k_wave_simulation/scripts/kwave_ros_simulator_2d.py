import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage.transform import resize
from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu, kspaceFirstOrder2D,kspaceFirstOrder2DC
import cv2
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped


from numba import njit, prange

import time


class kwave_process:
    def __init__(self):
        # 初始化元素
        rate = rospy.Rate(1)
        while True:

            self.msg = None
            self.probe_pose = None
            self.bridge = CvBridge()
            self.pub = rospy.Publisher('/kwave_image', Image, queue_size=10)
            self.orignal_pub = rospy.Publisher('/kwave_orginal_image', Image,queue_size=10)

            # 设置订阅者
            print("#########image_subscribe###############################")
            self.probe = rospy.Subscriber('/probe_pose',PoseStamped,self.probe_callback)
            self.sub = rospy.Subscriber('/slice_image', Image, self.volume_callback)
            rate.sleep()
            self.run()
    def probe_callback(self,msgs):
        self.probe_pose = msgs.pose.position

    def volume_callback(self, msgs):
        # 将 ROS 图像消息转换为 OpenCV 图像
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msgs, desired_encoding='passthrough')
            # cv2.imwrite('./test.png', cv_image)
        except CvBridgeError as e:
            print(e)
            return
        
        # 更新 self.msg
        self.msg = cv_image.astype(np.float64)
        
        # 直接在回调中运行模拟逻辑
        # self.run()
        # 
        
    def reconstruction(self,sensor_data):
        # Parameters
        Nx = 256  # Number of pixels in x direction (40mm / 0.2mm = 200)
        Ny = 256  # Number of pixels in y direction (40mm / 0.2mm = 200)
        dx = 0.2e-3  # x resolution [m]
        dy = 0.2e-3  # y resolution [m]
        c = 1500  # Speed of sound [m/s]
        fs = 25e6  # Sampling frequency [Hz]
        dt = 1/fs  # Time step [s]
        num_elements = 128  # Number of sensors
        pitch = 0.3e-3  # Array pitch [m]

        # sensor_data = cp.asarray(sensor_data)
        # Define imaging grid
        x_vec = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1) * dx  # x coordinate vector [m]
        y_vec = np.arange(0, Ny) * dy  # y coordinate vector [m] (starts from y=0)
        x, y = np.meshgrid(x_vec, y_vec)
        pixel_positions = np.column_stack((x.ravel(), y.ravel()))  # All pixel positions

        # Define sensor positions
        sensor_x = np.arange(-(num_elements-1)/2, (num_elements-1)/2 + 1) * pitch  # Sensor x coordinates [m]
        sensor_y = np.zeros(num_elements)  # Sensor y coordinates [m]
        sensor_positions = np.column_stack((sensor_x, sensor_y))

        # Initialize reconstructed image
        p_recon = np.zeros(Nx * Ny)

        # Assume sensor_data is available as a 2D numpy array
        # sensor_data = data[4, :, :]  # Shape: (time_samples, num_elements)
        # For this example, we'll create a dummy sensor_data array
        # Replace this with actual sensor data
        
        time1 = time.time()
        for pixel_idx in prange(Nx * Ny):
            
            # print("test")
            # Current pixel position
            pixel_pos = pixel_positions[pixel_idx, :]

            # Calculate distances from sensors to pixel
            distances = np.sqrt(np.sum((sensor_positions - pixel_pos) ** 2, axis=1))

            # Calculate delays (time step indices)
            delays = np.round(distances / c * fs).astype(int)
            # print("#####delays#####",np.shape(delays))
            # print("#####sensor_data##",np.shape(sensor_data))

            
            # Accumulate delayed signals
            # for sensor_idx in prange(num_elements):
                
            #     if 0 <= delays[sensor_idx] < 4096:
            #         p_recon[pixel_idx] += sensor_data[sensor_idx,delays[sensor_idx]]
    #         valid_indices = np.where((delays >= 0) & (delays < 4096))[0]

    # # 仅对有效索引进行操作
    #         for idx in valid_indices:
    #             p_recon[pixel_idx] += sensor_data[idx, delays[idx]]
            valid_mask = (delays >= 0) & (delays < 4096)
            valid_indices = np.where(valid_mask)[0]
            # 向量化累加
            p_recon[pixel_idx] += np.sum(sensor_data[valid_indices, delays[valid_indices]])
        time2 = time.time()
        print("###########reconstruction time###############",time2 - time1)

        # Reshape to image
        p_recon = p_recon.reshape(Nx, Ny)
        
        

        # Assuming p_recon is the reconstructed image from previous code
        # Initialize Hilbert transform result
        # p_recon_hilbert = np.zeros((Nx, Ny))

        # Apply Hilbert transform along each row (Y-axis) to extract envelope
        # for x_idx in range(Nx):
        #     analytic_signal = hilbert(p_recon[x_idx, :])
        #     p_recon_hilbert[x_idx, :] = np.abs(analytic_signal)  # Extract envelope

        # Normalize
        a = np.max(p_recon)
        b = np.min(p_recon)
        p_recon = (p_recon - b) / (a - b)*255

        # uint8_data = p_recon.astype(np.uint8)
        #########数据转回
        # uint8_data = np.asnumpy(uint8_data)
        # Normalize
        p_recon = p_recon / np.max(np.abs(p_recon))

        min_val, max_val = p_recon.min(), p_recon.max()
        if max_val != min_val:
            normalized = (p_recon - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(p_recon)
       
        uint8_data = normalized.astype(np.uint8)
        
        

        return uint8_data
    # @njit(parallel=True)
    # def reconstruction(self,sensor_data):
    #     # Initialize reconstructed image
    #     Nx = 256  # Number of pixels in x direction (40mm / 0.2mm = 200)
    #     Ny = 256  # Number of pixels in y direction (40mm / 0.2mm = 200)
    #     dx = 0.2e-3  # x resolution [m]
    #     dy = 0.2e-3  # y resolution [m]
    #     c = 1500  # Speed of sound [m/s]
    #     fs = 25e6  # Sampling frequency [Hz]
    #     dt = 1/fs  # Time step [s]
    #     num_elements = 128  # Number of sensors
    #     pitch = 0.3e-3  # Array pitch [m]
    #     p_recon = np.zeros(Nx * Ny)
    #     #     # Define imaging grid
    #     x_vec = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1) * dx  # x coordinate vector [m]
    #     y_vec = np.arange(0, Ny) * dy  # y coordinate vector [m] (starts from y=0)
    #     x, y = np.meshgrid(x_vec, y_vec)
    #     pixel_positions = np.column_stack((x.ravel(), y.ravel()))  # All pixel positions

    #     #     # Define sensor positions
    #     sensor_x = np.arange(-(num_elements-1)/2, (num_elements-1)/2 + 1) * pitch  # Sensor x coordinates [m]
    #     sensor_y = np.zeros(num_elements)  # Sensor y coordinates [m]
    #     sensor_positions = np.column_stack((sensor_x, sensor_y))

    #     for pixel_idx in prange(Nx * Ny):  # Parallelize this outer loop
    #         # Current pixel position
    #         pixel_pos = pixel_positions[pixel_idx, :]

    #         # Calculate distances from sensors to pixel
    #         distances = np.sqrt(np.sum((sensor_positions - pixel_pos) ** 2, axis=1))

    #         # Calculate delays (time step indices)
    #         delays = np.round(distances / c * fs).astype(int)

    #         # Accumulate delayed signals
    #         for sensor_idx in prange(num_elements):
    #             if 0 <= delays[sensor_idx] < 4096:
    #                 p_recon[pixel_idx] += sensor_data[sensor_idx, delays[sensor_idx]]

    #      #     # Reshape to image
    #     p_recon = p_recon.reshape(Nx, Ny)

        

    #     # Assuming p_recon is the reconstructed image from previous code
    #     # Initialize Hilbert transform result
    #     # p_recon_hilbert = np.zeros((Nx, Ny))

    #     # Apply Hilbert transform along each row (Y-axis) to extract envelope
    #     # for x_idx in range(Nx):
    #     #     analytic_signal = hilbert(p_recon[x_idx, :])
    #     #     p_recon_hilbert[x_idx, :] = np.abs(analytic_signal)  # Extract envelope

    #     # Normalize
    #     a = np.max(p_recon)
    #     b = np.min(p_recon)
    #     p_recon = (p_recon - b) / (a - b)*255

    #     uint8_data = p_recon.astype(np.uint8)
    #     # # Normalize
    #     p_recon = p_recon / np.max(np.abs(p_recon))

    #     min_val, max_val = p_recon.min(), p_recon.max()
    #     if max_val != min_val:
    #         normalized = (p_recon - min_val) / (max_val - min_val) * 255
    #     else:
    #         normalized = np.zeros_like(p_recon)
       
    #     uint8_data = normalized.astype(np.uint8)

    #     return uint8_data
    # def reconstruction(self, sensor_data,device):
    # # Parameters
    #     Nx = 256  # Number of pixels in x direction (40mm / 0.2mm = 200)
    #     Ny = 256  # Number of pixels in y direction (40mm / 0.2mm = 200)
    #     dx = 0.2e-3  # x resolution [m]
    #     dy = 0.2e-3  # y resolution [m]
    #     c = 1500  # Speed of sound [m/s]
    #     fs = 25e6  # Sampling frequency [Hz]
    #     dt = 1/fs  # Time step [s]
    #     num_elements = 128  # Number of sensors
    #     pitch = 0.3e-3  # Array pitch [m]

    #     # Define imaging grid (move to GPU)
        
    #     x_vec = torch.arange(-(Nx-1)/2, (Nx-1)/2 + 1) * dx  # x coordinate vector [m]
    #     y_vec = torch.arange(0, Ny) * dy  # y coordinate vector [m] (starts from y=0)
    #     x, y = torch.meshgrid(x_vec, y_vec,indexing='ij')
    #     pixel_positions = torch.column_stack((x.ravel(), y.ravel())).to(device)  # Move to GPU

    #     # Define sensor positions (move to GPU)
    #     sensor_x = torch.arange(-(num_elements-1)/2, (num_elements-1)/2 + 1) * pitch  # Sensor x coordinates [m]
    #     sensor_y = torch.zeros(num_elements)  # Sensor y coordinates [m]
    #     sensor_positions = torch.column_stack((sensor_x, sensor_y)).to(device)  # Move to GPU

    #     # Initialize reconstructed image
    #     p_recon = torch.zeros(Nx * Ny).to(device)  # Move to GPU

    #     # Assume sensor_data is available as a 2D numpy array and move it to GPU
    #     sensor_data = torch.tensor(sensor_data).to(device)
        

    #     for pixel_idx in prange(Nx * Ny):
    #         # Current pixel position
    #         pixel_pos = pixel_positions[pixel_idx, :]

    #         # Calculate distances from sensors to pixel
    #         distances = torch.sqrt(torch.sum((sensor_positions - pixel_pos) ** 2, dim=1))

    #         # Calculate delays (time step indices)
    #         delays = torch.round(distances / c * fs).long()

    #         # Accumulate delayed signals
    #         for sensor_idx in prange(num_elements):
    #             if 0 <= delays[sensor_idx] < sensor_data.shape[1]:  # Ensure within bounds
    #                 p_recon[pixel_idx] += sensor_data[sensor_idx, delays[sensor_idx]]
    #     print("$$$$$$$$")
    #     # Reshape to image
    #     p_recon = p_recon.reshape(Nx, Ny)

    #     # Normalize the image
    #     a = torch.max(p_recon)
    #     b = torch.min(p_recon)
    #     p_recon = (p_recon - b) / (a - b) * 255

    #     uint8_data = p_recon.to(torch.uint8)  # Convert to uint8
    #     print("#########")
    #     return uint8_data.cpu().numpy()  # Move back to CPU if necessary

    def compress_vertical(self,uint8_data, scale_factor=5):
   
    # 确保输入是uint8格式的numpy数组
        if not isinstance(uint8_data, np.ndarray) or uint8_data.dtype != np.uint8:
            raise ValueError("输入必须是uint8格式的numpy数组")
        
        # 获取原始尺寸
        height, width = uint8_data.shape[:2]
        
        # 计算压缩后的高度
        new_width = width // scale_factor
        
        # 使用cv2.resize进行纵向压缩
        # INTER_AREA适合缩放操作，能有效减少摩尔纹
        compressed_data = cv2.resize(uint8_data, (new_width, height), interpolation=cv2.INTER_AREA)
        
        # 确保输出是uint8格式
        compressed_data = compressed_data.astype(np.uint8)
        
        return compressed_data

    def run(self):
        # 检查 self.msg 是否为 None
        
        if self.msg is None:
            rospy.logwarn("No image data received yet, skipping simulation.")
            return
        
        
        # 定义 2D 网格
        dx = 0.2e-3  # [m]
        Nx, Ny = 256, 256
        kgrid = kWaveGrid([Nx, Ny], [dx, dx])
        # print('xxxxx',kgrid)
        # 定义介质属性
        medium = kWaveMedium(
            sound_speed=1540,  # [m/s]
            density=1000,      # [kg/m^3]
            alpha_coeff=0.75,  # [dB/(MHz^y cm)]
            alpha_power=1.5
        )
        kgrid.makeTime(medium.sound_speed)
        print(kgrid.dt*medium.sound_speed)
        # 定义传感器
        element_num = 128
        sensor_spacing = 0.3e-3
        sensor_y = -30
        # center_freq = 7.5e6
        # bandwidth = 65
        sensor_mask = np.zeros((Nx, Ny))
        sensor_start_x = round((Nx - (element_num - 1) * sensor_spacing / dx) / 2)
        sensor_points = np.round(np.arange(sensor_start_x, sensor_start_x + (element_num - 1) * sensor_spacing / dx + 1, sensor_spacing / dx)).astype(int)
        sensor_mask[sensor_points, sensor_y] = 1

        sensor = kSensor(mask=sensor_mask)
        sensor.record = ['p']
        # sensor.frequency_response = [center_freq, bandwidth]

        # 定义源
        source = kSource()
        source.p0 = np.zeros((Nx, Ny))
        source.p = None
        source.p_mask = np.zeros((Nx, Ny))
        if self.msg.shape == (Ny, Nx):
            source.p0 = self.msg/self.msg.max()
        else:
            source.p0 = resize(self.msg, (Nx, Ny))/self.msg.max()
        # plt.figure()
        # plt.imshow(source.p0 + sensor_mask, extent=[kgrid.x_vec[0]*1e3, kgrid.x_vec[-1]*1e3, kgrid.y_vec[-1]*1e3, kgrid.y_vec[0]*1e3], cmap='hot')
        
        # # plt.imshow(sensor_mask.T, extent=[kgrid.x_vec[0]*1e3, kgrid.x_vec[-1]*1e3, kgrid.y_vec[-1]*1e3, kgrid.y_vec[0]*1e3], cmap='hot')# plt.imshow(source.p0, extent=[kgrid.x_vec[0]*1e3, kgrid.x_vec[-1]*1e3, kgrid.y_vec[-1]*1e3, kgrid.y_vec[0]*1e3], cmap='hot')
        # # plt.plot(kgrid.x_vec[sensor_points]*1e3, np.ones_like(sensor_points)*kgrid.y_vec[sensor_y]*1e3, 'b*', markersize=5)
        # plt.title('Source and Detector Positions in 2D')
        # plt.xlabel('x (mm)')
        # plt.ylabel('y (mm)')
        # plt.axis('image')
        # plt.colorbar()
        # plt.show()
        # 创建仿真选项
        simulation_options = SimulationOptions(
            data_cast='single',
            save_to_disk=True,
            pml_size=20,
            smooth_c0=False
        )
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation = False
        )

        # 运行 2D GPU 仿真
        sensor_data = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)

        # 验证传感器数据
        if np.shape(sensor_data['p'])[1] != element_num:
            rospy.logerr(f"sensor_data['p'] has {np.shape(sensor_data['p'])[1]} sensors, expected {element_num}")
            return

        # 处理传感器数据
        combined_sensor_data = sensor_data['p'].T
        ############################################################
        print("reconstruction")
        reconstruction_data = self.reconstruction(combined_sensor_data)
        # cv2.imshow("./reconstruction",reconstruction_data)
        # cv2.imwrite("./reconstruction.png",reconstruction_data)



        ##############################################################
        data = combined_sensor_data
        min_val, max_val = data.min(), data.max()
        if max_val != min_val:
            normalized = (data - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(data)
       
        uint8_data = normalized.astype(np.uint8)
        uint8_data = self.compress_vertical(uint8_data,scale_factor=5)
        original_data = cv2.rotate(uint8_data, cv2.ROTATE_90_CLOCKWISE)
        # filename = "original_kwave_x" + str(self.probe_pose.x) + "_y" + str(self.probe_pose.y) + "_z" + str(self.probe_pose.z) + ".png" 
        # cv2.imwrite(filename,original_data)
        original_data = self.bridge.cv2_to_imgmsg(original_data, encoding="mono8")

        # 发布输出图像
        print("###############image publish######################")
        reconstruction_data = cv2.rotate(reconstruction_data, cv2.ROTATE_90_CLOCKWISE)
        # reconstruction_filename = "Target_kwave_reconstruction_x" + str(self.probe_pose.x) + "_y" + str(self.probe_pose.y) + "_z" + str(self.probe_pose.z) + ".png"
        # cv2.imwrite(reconstruction_filename,reconstruction_data)
        
        reconstruction_msg = self.bridge.cv2_to_imgmsg(reconstruction_data, encoding="mono8")
        
        self.pub.publish(reconstruction_msg)
        self.orignal_pub.publish(original_data)


if __name__ == '__main__':
    rospy.init_node('kwave_simulation')
    kwave_process()
    rospy.spin()