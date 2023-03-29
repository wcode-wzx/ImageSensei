import json

import cv2
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

    #-------------------图像几何变换------------------------
    def scale(self, scale_factor):
        # 缩放图像大小
        height, width = self.original_image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        resized_image = cv2.resize(self.original_image, (new_width, new_height))
        self.processed_image = resized_image
        return resized_image

    def translate(self, x, y):
        # 平移图像
        rows, cols = self.original_image.shape[:2]
        M = np.float32([[1, 0, x], [0, 1, y]])
        translated_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = translated_image
        return translated_image

    def rotate(self, angle):
        # 旋转图像
        rows, cols = self.original_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = rotated_image
        return rotated_image

    def affine(self, src_points, dst_points):
        # 仿射变换
        rows, cols = self.original_image.shape[:2]
        src_points = json.loads(src_points)
        dst_points = json.loads(dst_points)
        print(src_points, dst_points)
        M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
        transformed_image = cv2.warpAffine(self.original_image, M, (cols, rows))
        self.processed_image = transformed_image
        return transformed_image


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
