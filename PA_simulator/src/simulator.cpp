#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/opencv.hpp>
#include <vtkSmartPointer.h>
#include <vtkNIFTIImageReader.h>
#include <vtkSTLReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkVolume.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkImageReslice.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkAxesActor.h>
#include <vtkMath.h>
#include <vtkCommand.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkObjectFactory.h>
#include <vtkProperty.h>
#include <mutex>
#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include <sstream>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// Global mutex for latest_pose
std::mutex pose_mutex;

// Function to normalize quaternion
geometry_msgs::Quaternion normalize_quaternion(const geometry_msgs::Quaternion& q) {
    double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm == 0.0) {
        ROS_WARN("Quaternion norm is zero, returning original quaternion");
        return q;
    }
    geometry_msgs::Quaternion norm_q;
    norm_q.x = q.x / norm;
    norm_q.y = q.y / norm;
    norm_q.z = q.z / norm;
    norm_q.w = q.w / norm;
    return norm_q;
}

// Quaternion to rotation matrix (3x3)
void quaternion_to_matrix(const geometry_msgs::Quaternion& q, double R[3][3]) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    R[0][0] = 1 - 2 * y * y - 2 * z * z;
    R[0][1] = 2 * x * y - 2 * z * w;
    R[0][2] = 2 * x * z + 2 * y * w;
    R[1][0] = 2 * x * y + 2 * z * w;
    R[1][1] = 1 - 2 * x * x - 2 * z * z;
    R[1][2] = 2 * y * z - 2 * x * w;
    R[2][0] = 2 * x * z - 2 * y * w;
    R[2][1] = 2 * y * z + 2 * x * w;
    R[2][2] = 1 - 2 * x * x - 2 * y * y;
}

// Quaternion multiplication
geometry_msgs::Quaternion quaternion_mult(const geometry_msgs::Quaternion& q_a, const geometry_msgs::Quaternion& q_b) {
    geometry_msgs::Quaternion result;
    result.w = q_a.w * q_b.w - q_a.x * q_b.x - q_a.y * q_b.y - q_a.z * q_b.z;
    result.x = q_a.w * q_b.x + q_a.x * q_b.w + q_a.y * q_b.z - q_a.z * q_b.y;
    result.y = q_a.w * q_b.y - q_a.x * q_b.z + q_a.y * q_b.w + q_a.z * q_b.x;
    result.z = q_a.w * q_b.z + q_a.x * q_b.y - q_a.y * q_b.x + q_a.z * q_b.w;
    return result;
}

// Quaternion division (as in Python, effectively normalize if q_a == q_b)
geometry_msgs::Quaternion quaternion_div(const geometry_msgs::Quaternion& q_a, const geometry_msgs::Quaternion& q_b) {
    double norm = std::sqrt(q_b.x * q_b.x + q_b.y * q_b.y + q_b.z * q_b.z + q_b.w * q_b.w);
    if (norm == 0.0) norm = 1.0;
    geometry_msgs::Quaternion result;
    result.w = q_a.w / norm;
    result.x = q_a.x / norm;
    result.y = q_a.y / norm;
    result.z = q_a.z / norm;
    return result;
}

// Function to get slab volume
void get_slab_volume(vtkNIFTIImageReader* reader, vtkTransform* transform, double slab_thickness_mm, double spacing,
                     vtkMatrix4x4* axes, std::vector<float>& slab_data, int& n_slices, int& height, int& width) {
    n_slices = static_cast<int>(std::round(slab_thickness_mm / spacing));
    if (n_slices < 1) n_slices = 1;
    int half = n_slices % 2;

    slab_data.clear();
    std::vector<cv::Mat> slices;

    // Optional OMP parallel for if enabled
    // #pragma omp parallel for
    for (int i = -half; i <= half; ++i) {
        double offset = i * spacing;
        vtkSmartPointer<vtkMatrix4x4> axes_offset = vtkSmartPointer<vtkMatrix4x4>::New();
        axes_offset->DeepCopy(axes);
        double z_dir[3];
        z_dir[0] = axes->GetElement(0, 2);
        z_dir[1] = axes->GetElement(1, 2);
        z_dir[2] = axes->GetElement(2, 2);
        double norm_z = vtkMath::Norm(z_dir, 3);
        if (norm_z == 0.0) norm_z = 1.0;
        for (int j = 0; j < 3; ++j) {
            double trans = axes_offset->GetElement(j, 3) + offset * (z_dir[j] / norm_z);
            axes_offset->SetElement(j, 3, trans);
        }
        vtkSmartPointer<vtkImageReslice> local_reslice = vtkSmartPointer<vtkImageReslice>::New();
        local_reslice->SetInputConnection(reader->GetOutputPort());
        local_reslice->SetResliceAxes(axes_offset);
        local_reslice->SetOutputDimensionality(2);
        local_reslice->SetOutputExtent(0, 255, 0, 255, 0, 0);
        local_reslice->SetOutputSpacing(0.2, 0.2, 1.0);
        local_reslice->SetInterpolationModeToLinear();
        local_reslice->Update();
        vtkImageData* output = local_reslice->GetOutput();
        int dims[3];
        output->GetDimensions(dims);
        vtkDataArray* scalars = output->GetPointData()->GetScalars();
        int num_tuples = scalars->GetNumberOfTuples();

        cv::Mat img2d(dims[1], dims[0], CV_32F);
        float* ptr = img2d.ptr<float>(0);
        for (int k = 0; k < num_tuples; ++k) {
            ptr[k] = static_cast<float>(scalars->GetComponent(k, 0));
        }

        // Normalize to 0-255 uint8
        cv::normalize(img2d, img2d, 0, 255, cv::NORM_MINMAX);
        img2d.convertTo(img2d, CV_8U);

        // Rotate 90 degrees counterclockwise
        cv::rotate(img2d, img2d, cv::ROTATE_90_COUNTERCLOCKWISE);

        // ctSectorCutting
        int h = img2d.rows, w = img2d.cols;
        int cx = w / 2, cy = h / 2;
        int wx = 150, wy = 300;
        cv::Rect roi(cx - wx, cy - wy, 2 * wx, 2 * wy);
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > w || roi.y + roi.height > h) {
            // Handle out of bounds if necessary
            roi = cv::Rect(0, 0, w, h);
        }
        cv::Mat cropped = img2d(roi);
        cv::resize(cropped, cropped, cv::Size(256, 256));

        // Assuming serial, push to slices
        slices.push_back(cropped.clone());
    }

    // Now flatten all slices into slab_data as float
    height = 256;
    width = 256;
    for (const auto& slice : slices) {
        cv::Mat slice_float;
        slice.convertTo(slice_float, CV_32F);
        for (int row = 0; row < height; ++row) {
            const float* ptr = slice_float.ptr<float>(row);
            for (int col = 0; col < width; ++col) {
                slab_data.push_back(ptr[col]);
            }
        }
    }
}

class UltrasoundSimulator;

class TimerCallback : public vtkCommand {
public:
    static TimerCallback* New() {
        return new TimerCallback;
    }

    void SetSimulator(UltrasoundSimulator* sim) {
        this->sim = sim;
    }

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* callData);

private:
    UltrasoundSimulator* sim;
};

class UltrasoundSimulator {
public:
    UltrasoundSimulator() : latest_pose(nullptr), previous_pose(nullptr) {}

    void pose_callback(const geometry_msgs::PoseStampedConstPtr& msg) {
        geometry_msgs::PoseStamped pose = *msg;
        geometry_msgs::Point& p = pose.pose.position;
        const geometry_msgs::Quaternion& q = pose.pose.orientation;

        p.x += 2 * (q.x * q.z + q.w * q.y) * 0.17421;
        p.y += 2 * (q.y * q.z - q.w * q.x) * 0.17421;
        p.z += (1 - 2 * q.x * q.x - 2 * q.y * q.y) * 0.17421;

        visualization_msgs::MarkerArray marker_array;

        visualization_msgs::Marker position_marker;
        position_marker.header.frame_id = msg->header.frame_id;
        position_marker.header.stamp = ros::Time::now();
        position_marker.ns = "probe_position";
        position_marker.id = 0;
        position_marker.type = visualization_msgs::Marker::SPHERE;
        position_marker.action = visualization_msgs::Marker::ADD;
        position_marker.pose.position = p;
        position_marker.pose.orientation.w = 1.0;
        position_marker.scale.x = 0.05;
        position_marker.scale.y = 0.05;
        position_marker.scale.z = 0.05;
        position_marker.color.r = 0.0;
        position_marker.color.g = 1.0;
        position_marker.color.b = 0.0;
        position_marker.color.a = 1.0;
        marker_array.markers.push_back(position_marker);

        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = msg->header.frame_id;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "probe_position";
        text_marker.id = 1;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = p.x;
        text_marker.pose.position.y = p.y + 0.3;
        text_marker.pose.position.z = p.z + 0.1;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = 0.01;
        text_marker.color.r = 1.0;
        text_marker.color.g = 0.0;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0;
        std::stringstream ss;
        ss << pose.pose;
        text_marker.text = ss.str();
        marker_array.markers.push_back(text_marker);

        position_marker_pub.publish(marker_array);

        geometry_msgs::Quaternion q_normalized = normalize_quaternion(q);
        double R[3][3];
        quaternion_to_matrix(q_normalized, R);

        geometry_msgs::TransformStamped x_transform;
        x_transform.header.stamp = ros::Time::now();
        x_transform.header.frame_id = msg->header.frame_id;
        x_transform.child_frame_id = "probe_x_axis";
        x_transform.transform.translation.x = p.x + 0.1 * R[0][0];
        x_transform.transform.translation.y = p.y + 0.1 * R[1][0];
        x_transform.transform.translation.z = p.z + 0.1 * R[2][0];
        x_transform.transform.rotation.w = 1.0;
        tf_broadcaster.sendTransform(x_transform);

        geometry_msgs::TransformStamped y_transform;
        y_transform.header.stamp = ros::Time::now();
        y_transform.header.frame_id = msg->header.frame_id;
        y_transform.child_frame_id = "probe_y_axis";
        y_transform.transform.translation.x = p.x + 0.1 * R[0][1];
        y_transform.transform.translation.y = p.y + 0.1 * R[1][1];
        y_transform.transform.translation.z = p.z + 0.1 * R[2][1];
        y_transform.transform.rotation.w = 1.0;
        tf_broadcaster.sendTransform(y_transform);

        geometry_msgs::TransformStamped z_transform;
        z_transform.header.stamp = ros::Time::now();
        z_transform.header.frame_id = msg->header.frame_id;
        z_transform.child_frame_id = "probe_z_axis";
        x_transform.transform.translation.x = p.x + 0.1 * R[0][2];
        x_transform.transform.translation.y = p.y + 0.1 * R[1][2];
        x_transform.transform.translation.z = p.z + 0.1 * R[2][2];
        z_transform.transform.rotation.w = 1.0;
        tf_broadcaster.sendTransform(z_transform);

        std::lock_guard<std::mutex> lock(pose_mutex);
        latest_pose = boost::make_shared<geometry_msgs::PoseStamped>(pose);
    }

    void timer_callback() {
        geometry_msgs::PoseStampedPtr current_pose;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            if (!latest_pose) return;
            current_pose = latest_pose;
            latest_pose = nullptr;
        }

        clock_t start_time = clock();

        const geometry_msgs::Point& p = current_pose->pose.position;
        const geometry_msgs::Quaternion& q = current_pose->pose.orientation;

        double dx = 0.0, dy = 0.0, dz = 0.0;
        double q_real_norm[4] = {0.0, 0.0, 0.0, 0.0};
        if (previous_pose) {
            dx = p.x - previous_pose->pose.position.x;
            dy = p.y - previous_pose->pose.position.y;
            dz = p.z - previous_pose->pose.position.z;

            geometry_msgs::Quaternion previous_conj;
            previous_conj.x = -previous_pose->pose.orientation.x;
            previous_conj.y = -previous_pose->pose.orientation.y;
            previous_conj.z = -previous_pose->pose.orientation.z;
            previous_conj.w = previous_pose->pose.orientation.w;

            geometry_msgs::Quaternion q_real = quaternion_mult(previous_conj, q);
            geometry_msgs::Quaternion q_real_div = quaternion_div(q_real, q_real);
            // Round to 1 decimal
            q_real_norm[0] = std::round(q_real_div.w * 10.0) / 10.0;
            q_real_norm[1] = std::round(q_real_div.x * 10.0) / 10.0;
            q_real_norm[2] = std::round(q_real_div.y * 10.0) / 10.0;
            q_real_norm[3] = std::round(q_real_div.z * 10.0) / 10.0;
        }

        previous_pose = boost::make_shared<geometry_msgs::PoseStamped>(*current_pose);

        geometry_msgs::Quaternion q_normalized = normalize_quaternion(q);
        double R[3][3];
        quaternion_to_matrix(q_normalized, R);

        double p_mm[3] = {p.x * 1000.0, p.y * 1000.0, p.z * 1000.0};

        vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                matrix->SetElement(i, j, R[i][j]);
            }
            matrix->SetElement(i, 3, p_mm[i]);
        }

        transform->Translate(0, 0, extentT2[5] * spacingT2[2]);
        transform->Translate(dx, dy, dz);
        transform->RotateWXYZ(q_real_norm[0], q_real_norm[1], q_real_norm[2], q_real_norm[3]);
        transform->Translate(0, 0, -extentT2[5] * spacingT2[2]);

        probe_actor->SetUserMatrix(matrix);

        vtkSmartPointer<vtkMatrix4x4> T_probe_to_vessel = vtkSmartPointer<vtkMatrix4x4>::New();
        vtkMatrix4x4::Multiply4x4(T_vessel_to_base_inv, matrix, T_probe_to_vessel);

        vtkSmartPointer<vtkMatrix4x4> T_probe_to_vessel_y = vtkSmartPointer<vtkMatrix4x4>::New();
        vtkMatrix4x4::Multiply4x4(T_probe_to_vessel, R_y, T_probe_to_vessel_y);

        reslice->SetResliceAxes(T_probe_to_vessel_y);
        reslice->Update();

        vtkImageData* output = reslice->GetOutput();
        int dims[3];
        output->GetDimensions(dims);
        vtkDataArray* scalars = output->GetPointData()->GetScalars();
        int num_tuples = scalars->GetNumberOfTuples();
        int scalar_type = output->GetScalarType();

        cv::Mat image_array(dims[1], dims[0], CV_32F);
        float* img_ptr = image_array.ptr<float>(0);
        for (int k = 0; k < num_tuples; ++k) {
            img_ptr[k] = static_cast<float>(scalars->GetComponent(k, 0));
        }

        if (scalar_type == VTK_FLOAT) {
            double min_val, max_val;
            scalars->GetRange(&min_val);
            max_val = min_val + 1; // to avoid div0
            scalars->GetRange(&min_val, 0);
            scalars->GetRange(&max_val, 1);
            image_array = (image_array - min_val) / (max_val - min_val + 1e-6) * 255.0f;
            image_array.convertTo(image_array, CV_8U);
        } else if (scalar_type == VTK_UNSIGNED_CHAR) {
            // Assume already uint8
            image_array.convertTo(image_array, CV_8U);
        } else if (scalar_type == VTK_SHORT) {
            // Assume 16bit, but for mono8, normalize or cast
            double min_val, max_val;
            cv::minMaxLoc(image_array, &min_val, &max_val);
            image_array = (image_array - min_val) / (max_val - min_val + 1e-6) * 255.0f;
            image_array.convertTo(image_array, CV_8U);
        } else {
            ROS_WARN("Unsupported scalar type");
            return;
        }

        cv::imwrite("./original_us.png", image_array);
        // cv::minMaxLoc(image_array, &min_val);
        std::cout << "###########image_volume##########  " << image_array.size << std::endl;

        sensor_msgs::ImagePtr ros_image = cv_bridge::CvImage(std_msgs::Header(), "mono8", image_array).toImageMsg();
        image_pub.publish(ros_image);

        std::vector<float> slab_data;
        int n_slices_out, h, w;
        get_slab_volume(reader, transform, 4, spacingT2[2], T_probe_to_vessel_y, slab_data, n_slices_out, h, w);

        std_msgs::Float32MultiArray slab_msg;
        slab_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        slab_msg.layout.dim[0].label = "n_slices";
        slab_msg.layout.dim[0].size = n_slices_out;
        slab_msg.layout.dim[0].stride = h * w;
        slab_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        slab_msg.layout.dim[1].label = "h";
        slab_msg.layout.dim[1].size = h;
        slab_msg.layout.dim[1].stride = w;
        slab_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        slab_msg.layout.dim[2].label = "w";
        slab_msg.layout.dim[2].size = w;
        slab_msg.layout.dim[2].stride = 1;
        slab_msg.data = slab_data;
        slab_volume_pub.publish(slab_msg);
        std::cout << "#############slab_volume##########  " << n_slices_out << "x" << h << "x" << w << std::endl;

        render_window->Render();

        double exec_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
        std::cout << "elastic time: " << exec_time << std::endl;
    }

    void run() {
        ros::NodeHandle nh;

        position_marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/probe_position_marker", 10);
        image_pub = nh.advertise<sensor_msgs::Image>("/slice_image", 10);
        slab_volume_pub = nh.advertise<std_msgs::Float32MultiArray>("/slab_volume", 10);
        vessel_center_pub = nh.advertise<geometry_msgs::PoseStamped>("/vessel_center", 1, true);

        ros::Subscriber pose_sub = nh.subscribe("/probe_pose", 1, &UltrasoundSimulator::pose_callback, this);

        reader = vtkSmartPointer<vtkNIFTIImageReader>::New();
        reader->SetFileName("/home/thera/expert_simulator_pa_7_5/src/ct_to_us_simulator/vessel.nii");
        reader->Update();
        reader->GetOutput()->GetSpacing(spacingT2);
        reader->GetOutput()->GetExtent(extentT2);

        transform = vtkSmartPointer<vtkTransform>::New();
        transform->Translate(0, 0, 0);

        axes = vtkSmartPointer<vtkMatrix4x4>::New();
        axes->SetElement(0, 2, 1);
        axes->SetElement(1, 1, 1);
        axes->SetElement(2, 0, -1);

        vtkImageData* data = reader->GetOutput();
        double scalar_range[2];
        data->GetScalarRange(scalar_range);
        double bounds[6];
        data->GetBounds(bounds);
        double center_vessel[3] = {
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0
        };

        double p_desired[3] = {0.31414382092127224 * 1000, 0.20010206254387328 * 1000, 0.07398700643920156 * 1000};
        geometry_msgs::Quaternion q_desired;
        q_desired.x = 0.001554556123541251;
        q_desired.y = 0.9971732461084396;
        q_desired.z = -0.07503339723093785;
        q_desired.w = 0.0036179967938902266;
        geometry_msgs::Quaternion q_desired_normalized = normalize_quaternion(q_desired);
        double R_desired[3][3];
        quaternion_to_matrix(q_desired_normalized, R_desired);

        double R_y_neg90_np[3][3] = {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}};
        double R_desired_new[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R_desired_new[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    R_desired_new[i][j] += R_desired[i][k] * R_y_neg90_np[k][j];
                }
            }
        }

        double t[3];
        for (int i = 0; i < 3; ++i) {
            t[i] = p_desired[i];
            for (int j = 0; j < 3; ++j) {
                t[i] -= R_desired_new[i][j] * center_vessel[j];
            }
        }

        T_vessel_to_base = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T_vessel_to_base->SetElement(i, j, R_desired_new[i][j]);
            }
            T_vessel_to_base->SetElement(i, 3, t[i]);
        }

        T_vessel_to_base_inv = vtkSmartPointer<vtkMatrix4x4>::New();
        T_vessel_to_base_inv->DeepCopy(T_vessel_to_base);
        T_vessel_to_base_inv->Invert();

        R_x = vtkSmartPointer<vtkMatrix4x4>::New();
        R_x->SetElement(0, 0, 1);
        R_x->SetElement(1, 2, -1);
        R_x->SetElement(2, 1, 1);
        R_x->SetElement(3, 3, 1);

        R_y = vtkSmartPointer<vtkMatrix4x4>::New();
        R_y->SetElement(0, 2, 1);
        R_y->SetElement(1, 1, 1);
        R_y->SetElement(2, 0, -1);
        R_y->SetElement(3, 3, 1);

        R_z = vtkSmartPointer<vtkMatrix4x4>::New();
        R_z->SetElement(0, 1, -1);
        R_z->SetElement(1, 0, 1);
        R_z->SetElement(2, 2, 1);
        R_z->SetElement(3, 3, 1);

        geometry_msgs::PoseStamped vessel_pose;
        vessel_pose.header.frame_id = "base_link";
        vessel_pose.header.stamp = ros::Time::now();
        vessel_pose.pose.position.x = p_desired[0] / 1000.0;
        vessel_pose.pose.position.y = p_desired[1] / 1000.0;
        vessel_pose.pose.position.z = p_desired[2] / 1000.0;
        vessel_pose.pose.orientation = q_desired;
        vessel_center_pub.publish(vessel_pose);
        ROS_INFO("Published vessel center pose to /vessel_center");

        vtkSmartPointer<vtkSmartVolumeMapper> mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
        mapper->SetInputConnection(reader->GetOutputPort());
        vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
        volume->SetMapper(mapper);
        vtkSmartPointer<vtkVolumeProperty> prop = vtkSmartPointer<vtkVolumeProperty>::New();
        vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
        color->AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0);
        color->AddRGBPoint(scalar_range[0] + 0.25 * (scalar_range[1] - scalar_range[0]), 0.3, 0.3, 0.3);
        color->AddRGBPoint(scalar_range[0] + 0.5 * (scalar_range[1] - scalar_range[0]), 0.6, 0.6, 0.6);
        color->AddRGBPoint(scalar_range[0] + 0.75 * (scalar_range[1] - scalar_range[0]), 0.9, 0.9, 0.9);
        color->AddRGBPoint(scalar_range[1], 1.0, 1.0, 1.0);
        prop->SetColor(color);
        vtkSmartPointer<vtkPiecewiseFunction> opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
        opacity->AddPoint(scalar_range[0], 0.0);
        opacity->AddPoint(scalar_range[0] + 0.08 * (scalar_range[1] - scalar_range[0]), 0.5);
        opacity->AddPoint(scalar_range[1], 1.0);
        prop->SetScalarOpacity(opacity);
        prop->SetAmbient(0.3);
        volume->SetProperty(prop);
        volume->SetUserMatrix(T_vessel_to_base);

        vtkSmartPointer<vtkSTLReader> stl_reader = vtkSmartPointer<vtkSTLReader>::New();
        stl_reader->SetFileName("/home/thera/expert_simulator_pa_7_5/src/linescan_mm.STL");
        stl_reader->Update();
        vtkSmartPointer<vtkPolyDataMapper> probe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        probe_mapper->SetInputConnection(stl_reader->GetOutputPort());
        probe_actor = vtkSmartPointer<vtkActor>::New();
        probe_actor->SetMapper(probe_mapper);
        probe_actor->RotateX(180);
        probe_actor->GetProperty()->SetColor(1, 0, 0);

        vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
        renderer->AddVolume(volume);
        renderer->AddActor(probe_actor);
        renderer->SetBackground(0, 0, 0);

        render_window = vtkSmartPointer<vtkRenderWindow>::New();
        render_window->AddRenderer(renderer);
        render_window->SetSize(800, 600);

        vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        interactor->SetRenderWindow(render_window);

        vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
        style->SetMotionFactor(2.0);
        interactor->SetInteractorStyle(style);

        interactor->Initialize();

        renderer->ResetCamera();

        reslice = vtkSmartPointer<vtkImageReslice>::New();
        reslice->SetInputConnection(reader->GetOutputPort());
        reslice->SetOutputDimensionality(2);
        reslice->SetOutputExtent(0, 255, 0, 255, 0, 0);
        reslice->SetOutputSpacing(0.2, 0.2, 1);

        vtkSmartPointer<TimerCallback> timer_cb = vtkSmartPointer<TimerCallback>::New();
        timer_cb->SetSimulator(this);
        interactor->AddObserver(vtkCommand::TimerEvent, timer_cb);
        interactor->CreateRepeatingTimer(10);

        ros::AsyncSpinner spinner(1);
        spinner.start();

        interactor->Start();
    }

private:
    typedef boost::shared_ptr<geometry_msgs::PoseStamped> PoseStampedPtr;
    PoseStampedPtr latest_pose;
    PoseStampedPtr previous_pose;
    ros::Publisher position_marker_pub;
    tf2_ros::TransformBroadcaster tf_broadcaster;
    ros::Publisher image_pub;
    ros::Publisher slab_volume_pub;
    ros::Publisher vessel_center_pub;

    vtkSmartPointer<vtkNIFTIImageReader> reader;
    vtkSmartPointer<vtkTransform> transform;
    vtkSmartPointer<vtkMatrix4x4> axes;
    vtkSmartPointer<vtkMatrix4x4> T_vessel_to_base;
    vtkSmartPointer<vtkMatrix4x4> T_vessel_to_base_inv;
    vtkSmartPointer<vtkMatrix4x4> R_x;
    vtkSmartPointer<vtkMatrix4x4> R_y;
    vtkSmartPointer<vtkMatrix4x4> R_z;
    vtkSmartPointer<vtkActor> probe_actor;
    vtkSmartPointer<vtkRenderWindow> render_window;
    vtkSmartPointer<vtkImageReslice> reslice;
    double spacingT2[3];
    int extentT2[6];
};

void TimerCallback::Execute(vtkObject* caller, unsigned long eventId, void* callData) {
    if (eventId == vtkCommand::TimerEvent) {
        sim->timer_callback();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ultrasound_simulator");
    UltrasoundSimulator sim;
    sim.run();
    return 0;
}