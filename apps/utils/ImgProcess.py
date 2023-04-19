import json

import cv2
import math
import numpy as np
import base64


class ImageProcessor:
    def __init__(self):
        image = cv2.imread('apps/static/sample.png')
        self.original_image = image
        self.processed_image = image

    def overwrite_original_image(self):
        self.original_image = self.processed_image

    def set_processed_image(self):
        image = cv2.imread('apps/static/sample.png')
        self.processed_image = image

    def set_original_image(self, img):
        self.original_image = img

    def get_original_image(self):
        return self.original_image

    def get_processed_image(self):
        return self.processed_image

    def enhance_contrast(self, alpha, beta):
        """
        其中，enhance_contrast函数使用了OpenCV的convertScaleAbs函数对图像进行对比度增强；
        :param alpha:
        :param beta:
        :return:
        """
        self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)

    def histogram_equalization(self):
        """
        histogram_equalization函数使用了OpenCV的equalizeHist函数对图像进行直方图均衡化；
        :return:
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.equalizeHist(gray_image)

    def filter_image(self, kernel):
        """
        filter_image函数使用了OpenCV的filter2D函数对图像进行滤波；
        :param kernel:
        :return:
        """
        self.processed_image = cv2.filter2D(self.original_image, -1, kernel)

    def sharpen_image(self):
        """
        sharpen_image函数使用了自定义的锐化卷积核对图像进行锐化；
        :return:
        """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.processed_image = cv2.filter2D(self.original_image, -1, kernel)

    def morphological_processing(self, operation, kernel):
        """
        morphological_processing函数使用了OpenCV的形态学处理函数对图像进行腐蚀、膨胀、开运算和闭运算操作；
        :param operation:
        :param kernel:
        :return:
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        if operation == 'erosion':
            self.processed_image = cv2.erode(self.original_image, kernel)
        elif operation == 'dilation':
            self.processed_image = cv2.dilate(self.original_image, kernel)
        elif operation == 'opening':
            self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)

    def edge_detection(self):
        """
        edge_detection函数使用了OpenCV的Canny函数对图像进行边缘检测；
        :return:
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.Canny(gray_image, 100, 200)

    def binarize_image(self, threshold):
        """
        binarize_image函数使用了OpenCV的threshold函数对图像进行二值化处理
        :param threshold:
        :return:
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # -------------------图像几何变换------------------------
    def scale(self, scale_factor):
        """
        缩放图像大小
        :param scale_factor:
        :return:
        """
        height, width = self.original_image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        resized_image = cv2.resize(self.original_image, (new_width, new_height))
        self.processed_image = resized_image
        return resized_image

    def translate(self, x, y):
        """
        平移图像
        :param x:
        :param y:
        :return:
        """
        rows, cols = self.original_image.shape[:2]
        M = np.float32([[1, 0, x], [0, 1, y]])
        translated_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = translated_image
        return translated_image

    def rotate(self, angle):
        """
        旋转图像
        :param angle:
        :return:
        """
        rows, cols = self.original_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = rotated_image
        return rotated_image

    def affine(self, src_points, dst_points):
        """
        仿射变换
        :param src_points:
        :param dst_points:
        :return:
        """
        rows, cols = self.original_image.shape[:2]
        src_points = json.loads(src_points)
        dst_points = json.loads(dst_points)
        M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
        transformed_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = transformed_image
        return transformed_image

    # -------------------图像阈值------------------------
    def threshold(self, threshold_value, max_value, threshold_type):
        """

        :param threshold_value:
        :param max_value:
        :param threshold_type:
        :return:
        """
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
        _, thresh_img = cv2.threshold(gray_img, int(threshold_value), int(max_value), eval(threshold_type))  # 应用全局阈值
        self.processed_image = thresh_img
        return thresh_img

    def adaptive_threshold(self, block_size, c, threshold_type):
        """

        :param block_size:
        :param c:
        :param threshold_type:
        :return:
        """
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, threshold_type, cv2.THRESH_BINARY, block_size, c)  # 应用自适应阈值
        self.processed_image = thresh_img
        return thresh_img

    def binary(self):
        """
        二值化
        :return:
        """
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
        _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)  # 应用全局阈值
        self.processed_image = thresh_img
        return thresh_img

    def grayscale(self):
        """
        灰度化
        :return:
        """
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
        self.processed_image = gray_img
        return gray_img

    # -------------------平滑图像------------------------
    def set_kernel_size(self, kernel_size):
        """设置kernel_size

        Args:
            kernel_size (int): kernel_size大小
        """
        self.kernel_size = kernel_size

    def mean_blur(self):
        """均值模糊"""
        self.processed_image = cv2.blur(self.original_image, (self.kernel_size, self.kernel_size))

    def gaussian_blur(self):
        """高斯模糊"""
        self.processed_image = cv2.GaussianBlur(self.original_image, (self.kernel_size, self.kernel_size), 0)

    def median_blur(self):
        """中值滤波"""
        self.processed_image = cv2.medianBlur(self.original_image, self.kernel_size)

    def bilateral_filter(self):
        """双边滤波"""
        self.processed_image = cv2.bilateralFilter(self.original_image, self.kernel_size, 75, 75)

    # -------------------形态转换------------------------
    def set_kernel(self, kernel_size: int):
        """
        构造函数

        :param kernel_size: 核大小
        """
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def erode(self):
        """
        腐蚀操作
        """
        self.processed_image = cv2.erode(self.original_image, self.kernel, iterations=1)

    def dilate(self):
        """
        膨胀操作
        """
        self.processed_image = cv2.dilate(self.original_image, self.kernel, iterations=1)

    def opening(self):
        """
        开运算操作
        """
        self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, self.kernel)

    def closing(self):
        """
        闭运算操作
        """
        self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, self.kernel)

    def gradient(self):
        """
        形态梯度操作
        """
        self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_GRADIENT, self.kernel)

    def tophat(self):
        """
        顶帽操作
        """
        self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_TOPHAT, self.kernel)

    def blackhat(self):
        """
        黑帽操作
        """
        self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_BLACKHAT, self.kernel)

    # -------------------边缘检测------------------------
    def edge_detection(self, minVal, maxVal):
        """
        边缘检测函数

        :param image: 输入图像
        :return: 边缘检测后的图像
        """
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # 使用高斯滤波平滑图像
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 使用 Canny 算法进行边缘检测
        self.processed_image = cv2.Canny(blurred, minVal, maxVal)

    # -------------------模板匹配------------------------
    def template_matching(self, template: np.ndarray, method: str) -> np.ndarray:
        """
        使用模板匹配在图像中查找模板，并返回带有所有匹配位置框的图像。

        Args:
            img: 待搜索的图像，应该是一个灰度图像。
            template: 要搜索的模板，应该是一个灰度图像。

        Returns:
            带有所有匹配位置框的图像。
        """
        original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        template_image = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 获取模板的宽度和高度
        w, h = template_image.shape[::-1]
        print(w, h)

        # 使用模板匹配算法在图像中搜索模板
        res = cv2.matchTemplate(original_image, template_image, eval(method))

        # 找到所有匹配位置
        loc = np.where(res >= 0.8)

        # 在原始图像上绘制所有匹配位置框
        temp = self.original_image.copy()
        for pt in zip(*loc[::-1]):
            cv2.rectangle(temp, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

        # 返回带有所有匹配位置框的图像
        self.processed_image = temp
        return self.processed_image

    # -------------------霍夫变换------------------------
    def detect_lines(self, rho_res=1, threshold=200):
        """
        霍夫线变换
        :param rho_res:
        :param threshold:
        :return:
        """
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, rho_res, np.pi / 180, threshold)
        img = self.original_image.copy()
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        self.processed_image = img
        return img

    def detect_circles(self, param1=50, param2=30, minRadius=0, maxRadius=0):
        """
        霍夫圆变换
        :param param1:
        :param param2:
        :param minRadius:
        :param maxRadius:
        :return:
        """
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        self.processed_image = cimg
        return cimg




def img_to_base64(img_array):
    """
    传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    :param img_array:
    :return:
    """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1]  # 用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes()  # 转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 转换为base64
    return base64_str


def base64_to_img(base64_str):
    """
    传入为RGB格式下的base64，传出为RGB格式的numpy矩阵
    :param base64_str:
    :return:
    """
    byte_data = base64.b64decode(base64_str)  # 将base64转换为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")  # 二进制转换为一维数组
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)  # 用cv2解码为三通道矩阵
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # BGR2RGB
    return img_array
