import numpy as np
import nibabel as nib
import cv2
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support


from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2


def sliding_average(new_value, history, window_size):
    """
    滑动平均计算
    """
    history.append(new_value)
    if len(history) > window_size:
        history.popleft()
    return np.mean(history)

def ctSectorCutting(image):
    """
    简化：仅裁剪中心区域并缩放为600x300
    """
    h, w = image.shape
    cx, cy = w // 2, h // 2
    wx, wy = 150, 300
    image = image[cy - wy:cy + wy, cx - wx:cx + wx]
    image = cv2.resize(image, (256,256))#######重新定义为256*256
    return image

def voxel2array(img):
    """
    将 VTK 图像数据转换为 NumPy 数组。

    参数：
    - img (vtkImageData): 输入的 VTK 图像数据。

    返回：
    - image_arr (numpy.ndarray): 转换后的 NumPy 数组。

    说明：
    该函数将输入的 VTK 图像数据转换为 NumPy 数组，并返回转换后的二维数组。

    """

    from vtk.util import numpy_support

    # 获取体数据的维度: (x, y, z)
    dims = img.GetDimensions()  # (x, y, z)
    # 从 VTK 图像数据提取原始一维数据
    sc = img.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(sc)

    # 重新排列为 (z, y, x)，注意 VTK 存储顺序与 numpy 不同
    arr = arr.reshape(dims[::-1])
    # squeeze 去除单维，比如 z=1 时得到 (y, x) 的二维数组
    arr = np.squeeze(arr)
    return arr

def get_slab_volume(image_reader, transform, slab_thickness_mm, spacing, axes):
    """
    获取厚度为slab_thickness_mm的三维切片体（未合成为2D），可直接作为三维网络输入。
    
    参数:
    image_reader: VTK NIFTI/Meta/其他图像读取器，已调用Update()
    transform:   vtkTransform, 切片中心（探头）位姿
    slab_thickness_mm: float, slab体厚度（单位：mm）
    spacing:     float, 沿切片法向的体素间距（单位：mm）
    axes:        vtkMatrix4x4, 切片方向矩阵（见你的reslice.SetResliceAxes用法）

    返回:
    slab_volume: np.ndarray, shape=(n_slices, h, w)，三维切片体
    """
    import numpy as np
    import vtkmodules.all as vtk

    # 计算需要采集的层数
    n_slices = int(round(slab_thickness_mm / spacing))
    if n_slices < 1: n_slices = 1
    half = n_slices // 2

    slices = []
    for i in range(-half, half+1):
        # 对transform进行深拷贝
        slice_transform = vtk.vtkTransform()
        slice_transform.DeepCopy(transform)
        # 假定法向为z轴（如果不是，请替换下面的Translate参数）
        slice_transform.Translate(0, 0, i * spacing)
        reslice = vtk.vtkImageReslice()
        reslice.SetInputConnection(image_reader.GetOutputPort())
        reslice.SetResliceTransform(slice_transform)
        reslice.SetResliceAxes(axes)
        reslice.SetOutputDimensionality(2)
        reslice.SetInterpolationModeToLinear()
        reslice.Update()
        img2d = voxel2array(reslice.GetOutput())

        # 垂直翻转矩阵
        img2d = cv2.normalize(img2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 旋转90度（逆时针）
        img2d = cv2.rotate(img2d, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img2d = ctSectorCutting(img2d) 

        slices.append(img2d)

    slab_volume = np.stack(slices, axis=0)  # (n_slices, h, w)
    return slab_volume

