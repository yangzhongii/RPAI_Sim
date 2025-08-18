#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <H5Cpp.h>
#include <omp.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdlib>

class KWaveProcess {
private:
    ros::NodeHandle nh_;
    ros::Subscriber probe_sub_;
    ros::Subscriber image_sub_;
    ros::Publisher kwave_pub_;
    ros::Publisher original_pub_;
    // cv_bridge::CvBridge bridge_;
    cv::Mat msg_; // Current image
    geometry_msgs::Pose probe_pose_;
    bool has_image_;

    // Simulation parameters
    const int Nx = 256;
    const int Ny = 256;
    const double dx = 0.2e-3; // [m]
    const double dy = 0.2e-3; // [m]
    const double sound_speed = 1540.0; // [m/s]
    const double density = 1000.0; // [kg/m^3]
    const double alpha_coeff = 0.75; // [dB/(MHz^y cm)]
    const double alpha_power = 1.5;
    const int element_num = 128;
    const double sensor_spacing = 0.3e-3; // [m]
    const int sensor_y = -30;
    const double fs = 25e6; // [Hz]
    const double dt = 1.0 / fs; // [s]
    const double pitch = 0.3e-3; // [m]

public:
    KWaveProcess() : nh_("~"), has_image_(false) {
        std::cout << "KWaveProcess initialized with parameters:" << std::endl;
        probe_sub_ = nh_.subscribe("/probe_pose", 10, &KWaveProcess::probeCallback, this);
        image_sub_ = nh_.subscribe("/slice_image", 10, &KWaveProcess::volumeCallback, this);
        kwave_pub_ = nh_.advertise<sensor_msgs::Image>("/kwave_image", 10);
        original_pub_ = nh_.advertise<sensor_msgs::Image>("/kwave_orginal_image", 10);
        std::cout << "Nx: " << Nx << ", Ny: " << Ny << ", dx: " << dx << ", dy: " << dy << std::endl;
    }

    void probeCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        std::cout << "Received probe pose: " << msg->pose.position.x << ", "
                  << msg->pose.position.y << ", " << msg->pose.position.z << std::endl;
        probe_pose_ = msg->pose;
    }

    void volumeCallback(const sensor_msgs::Image::ConstPtr& msg) {
        try {
            msg_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16)->image;
            
            msg_.convertTo(msg_, CV_64F); // Convert to double for processing
            std::cout << "Received image of size: " << std::endl;
            has_image_ = true;
            run();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("CvBridge Error: %s", e.what());
        }
        
    }

    cv::Mat reconstruct(const Eigen::MatrixXd& sensor_data) {
        // Parameters
        std::cout << "Reconstructing with parameters:" << std::endl;
        const int time_samples = sensor_data.cols();
        Eigen::VectorXd x_vec = Eigen::VectorXd::LinSpaced(Nx, -(Nx-1)/2.0, (Nx-1)/2.0) * dx;
        Eigen::VectorXd y_vec = Eigen::VectorXd::LinSpaced(Ny, 0, Ny-1) * dy;
        Eigen::MatrixXd pixel_positions(Nx * Ny, 2);
        for (int i = 0; i < Ny; ++i) {
            for (int j = 0; j < Nx; ++j) {
                pixel_positions(i * Nx + j, 0) = x_vec(j);
                pixel_positions(i * Nx + j, 1) = y_vec(i);
            }
        }

        Eigen::VectorXd sensor_x = Eigen::VectorXd::LinSpaced(element_num, -(element_num-1)/2.0, (element_num-1)/2.0) * pitch;
        Eigen::VectorXd sensor_y = Eigen::VectorXd::Zero(element_num);
        Eigen::MatrixXd sensor_positions(element_num, 2);
        sensor_positions.col(0) = sensor_x;
        sensor_positions.col(1) = sensor_y;

        Eigen::VectorXd p_recon = Eigen::VectorXd::Zero(Nx * Ny);
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int pixel_idx = 0; pixel_idx < Nx * Ny; ++pixel_idx) {
            Eigen::Vector2d pixel_pos = pixel_positions.row(pixel_idx);
            Eigen::VectorXd distances = (sensor_positions.rowwise() - pixel_pos.transpose()).rowwise().norm();
            Eigen::VectorXi delays = (distances / sound_speed * fs).array().round().cast<int>();
            double sum = 0.0;
            for (int sensor_idx = 0; sensor_idx < element_num; ++sensor_idx) {
                if (delays(sensor_idx) >= 0 && delays(sensor_idx) < time_samples) {
                    sum += sensor_data(sensor_idx, delays(sensor_idx));
                }
            }
            p_recon(pixel_idx) = sum;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Reconstruction time: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

        // Reshape and normalize
        cv::Mat p_recon_mat(Ny, Nx, CV_64F, p_recon.data());
        double min_val, max_val;
        cv::minMaxLoc(p_recon_mat, &min_val, &max_val);
        cv::Mat normalized;
        if (max_val != min_val) {
            normalized = (p_recon_mat - min_val) / (max_val - min_val) * 255;
        } else {
            normalized = cv::Mat::zeros(p_recon_mat.size(), p_recon_mat.type());
        }
        cv::Mat uint8_data;
        normalized.convertTo(uint8_data, CV_8U);
        return uint8_data;
    }

    cv::Mat compressVertical(const cv::Mat& uint8_data, int scale_factor = 5) {
        if (uint8_data.type() != CV_8U) {
            throw std::runtime_error("Input must be uint8 format");
        }
        int height = uint8_data.rows;
        int width = uint8_data.cols;
        int new_width = width / scale_factor;
        cv::Mat compressed_data;
        cv::resize(uint8_data, compressed_data, cv::Size(new_width, height), 0, 0, cv::INTER_AREA);
        return compressed_data;
    }

void writeHDF5Input(const cv::Mat& source_p0, const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    hsize_t dims[2];

    // Grid parameters
    dims[0] = 1;
    H5::DataSpace scalar_space(1, dims);
    H5::DataSet dataset = file.createDataSet("Nx", H5::PredType::NATIVE_INT, scalar_space);
    dataset.write(&Nx, H5::PredType::NATIVE_INT);
    dataset = file.createDataSet("Ny", H5::PredType::NATIVE_INT, scalar_space);
    dataset.write(&Ny, H5::PredType::NATIVE_INT);
    dataset = file.createDataSet("dx", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&dx, H5::PredType::NATIVE_DOUBLE);
    dataset = file.createDataSet("dy", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&dy, H5::PredType::NATIVE_DOUBLE);
    double t_end = (Nx * dx) / sound_speed * 2;
    dataset = file.createDataSet("t_end", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&t_end, H5::PredType::NATIVE_DOUBLE);
    dataset = file.createDataSet("dt", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&dt, H5::PredType::NATIVE_DOUBLE);

    // Medium parameters
    dataset = file.createDataSet("sound_speed", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&sound_speed, H5::PredType::NATIVE_DOUBLE);
    dataset = file.createDataSet("density", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&density, H5::PredType::NATIVE_DOUBLE);
    dataset = file.createDataSet("alpha_coeff", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&alpha_coeff, H5::PredType::NATIVE_DOUBLE);
    dataset = file.createDataSet("alpha_power", H5::PredType::NATIVE_DOUBLE, scalar_space);
    dataset.write(&alpha_power, H5::PredType::NATIVE_DOUBLE);

    // Source (p0)
    dims[0] = Nx; dims[1] = Ny;
    H5::DataSpace grid_space(2, dims);
    dataset = file.createDataSet("p0", H5::PredType::NATIVE_DOUBLE, grid_space);
    dataset.write(source_p0.data, H5::PredType::NATIVE_DOUBLE);

    // Sensor mask
    Eigen::MatrixXi sensor_mask = Eigen::MatrixXi::Zero(Nx, Ny);
    int sensor_start_x = round((Nx - (element_num - 1) * sensor_spacing / dx) / 2);
    std::vector<int> sensor_points;
    for (int i = 0; i < element_num; ++i) {
        int x = round(sensor_start_x + i * (sensor_spacing / dx));
        if (x >= 0 && x < Nx) {
            sensor_points.push_back(x);
        }
    }
    int sensor_y_pos = 0;
    for (int x : sensor_points) {
        if (x >= 0 && x < Nx && sensor_y_pos >= 0 && sensor_y_pos < Ny) {
            sensor_mask(x, sensor_y_pos) = 1;
        }
    }
    dataset = file.createDataSet("sensor_mask", H5::PredType::NATIVE_INT, grid_space);
    dataset.write(sensor_mask.data(), H5::PredType::NATIVE_INT);

    // Simulation options
    int pml_size = 20;
    dataset = file.createDataSet("pml_size", H5::PredType::NATIVE_INT, scalar_space);
    dataset.write(&pml_size, H5::PredType::NATIVE_INT);
    std::string data_cast = "single";
    hsize_t str_dims[1] = {1};
    H5::DataSpace str_space(1, str_dims);
    H5::StrType str_type(H5::PredType::C_S1, 10);
    dataset = file.createDataSet("data_cast", str_type, str_space);
    dataset.write(data_cast.c_str(), str_type);
}
    Eigen::MatrixXd readHDF5SensorData(const std::string& filename) {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("p");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        Eigen::MatrixXd sensor_data(dims[0], dims[1]);
        dataset.read(sensor_data.data(), H5::PredType::NATIVE_DOUBLE);
        return sensor_data;
    }

    void run() {
        if (!has_image_) {
            ROS_WARN("No image data received yet, skipping simulation.");
            return;
        }

        // Prepare source
        cv::Mat source_p0;
        if (msg_.rows == Ny && msg_.cols == Nx) {
            source_p0 = msg_ / cv::norm(msg_, cv::NORM_INF);
        } else {
            cv::resize(msg_, source_p0, cv::Size(Nx, Ny), 0, 0, cv::INTER_AREA);
            source_p0 = source_p0 / cv::norm(source_p0, cv::NORM_INF);
        }

        // Write HDF5 input file
        std::cout << "Writing HDF5 input file" << std::endl;
        std::string input_file = "/tmp/kwave_input.h5";
        std::string output_file = "/tmp/kwave_output.h5";
        writeHDF5Input(source_p0, input_file);
        std::cout << "HDF5 input file written to: " << input_file << std::endl;

        // Run kspaceFirstOrder-CUDA
        std::cout << "Running kspaceFirstOrder-CUDA with input file: " << input_file << std::endl;
        std::string command = "/home/thera/simulator_photoacoustic/devel/lib/kwave_simulation/kspaceFirstOrder-CUDA -i " + input_file + " -o " + output_file;
        int ret = std::system(command.c_str());
        if (ret != 0) {
            ROS_ERROR("kspaceFirstOrder-CUDA failed with return code %d", ret);
            return;
        }

        // Read sensor data
        Eigen::MatrixXd sensor_data = readHDF5SensorData(output_file);
        if (sensor_data.cols() != element_num) {
            ROS_ERROR("sensor_data has %ld sensors, expected %d", sensor_data.cols(), element_num);
            return;
        }

        // Reconstruction
        ROS_INFO("Starting reconstruction");
        cv::Mat reconstruction_data = reconstruct(sensor_data);

        // Process sensor data for original image
        cv::Mat data(sensor_data.rows(), sensor_data.cols(), CV_64F, sensor_data.data());
        double min_val, max_val;
        cv::minMaxLoc(data, &min_val, &max_val);
        cv::Mat normalized;
        if (max_val != min_val) {
            normalized = (data - min_val) / (max_val - min_val) * 255;
        } else {
            normalized = cv::Mat::zeros(data.size(), data.type());
        }
        cv::Mat uint8_data;
        normalized.convertTo(uint8_data, CV_8U);
        uint8_data = compressVertical(uint8_data, 5);
        cv::rotate(uint8_data, uint8_data, cv::ROTATE_90_CLOCKWISE);

        // Publish results
        sensor_msgs::ImagePtr original_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", uint8_data).toImageMsg();
        cv::rotate(reconstruction_data, reconstruction_data, cv::ROTATE_90_CLOCKWISE);
        sensor_msgs::ImagePtr recon_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", reconstruction_data).toImageMsg();
        kwave_pub_.publish(recon_msg);
        original_pub_.publish(original_msg);
        ROS_INFO("Published kwave and original images");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kwave_simulation");
    KWaveProcess kwave_process;
    ros::Rate rate(1);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}