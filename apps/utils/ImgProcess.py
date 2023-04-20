import json

import cv2
import numpy as np
import base64
from imutils.perspective import four_point_transform


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

    # 人脸识别
    def face_recognition(self, scaleFactor=1.1, minNeighbors=3):
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        img = self.original_image.copy()

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self.processed_image = img
        if len(faces) == 0:
            return False
        else:
            return True

    # 答题卡识别
    def imgBrightness(self, img1, c, b):
        rows, cols = img1.shape
        blank = np.zeros([rows, cols], img1.dtype)
        rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
        return rst

    def answer_sheet_identification(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # 高斯滤波
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # 增强亮度
        blurred = self.imgBrightness(blurred, 1.5, 4)

        # 自适应二值化
        blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
        '''
        adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
            第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
            第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
            第四个参数threshold_type  指取阈值类型：必须是下者之一  
                                • CV_THRESH_BINARY,
                                • CV_THRESH_BINARY_INV
            第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
            第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
        '''
        blurred = cv2.copyMakeBorder(blurred, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # canny边缘检测
        edged = cv2.Canny(blurred, 0, 255)
        cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        docCnt = []
        count = 0
        # 确保至少有一个轮廓被找到
        if len(cnts) > 0:
            # 将轮廓按照大小排序
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 对排序后的轮廓进行循环处理
        for c in cnts:
            # 获取近似的轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
            if len(approx) == 4:
                docCnt.append(approx)
                count += 1
                if count == 3:
                    break

        # 四点变换，划出选择题区域
        paper = four_point_transform(self.original_image, np.array(docCnt[0]).reshape(4, 2))
        warped = four_point_transform(gray, np.array(docCnt[0]).reshape(4, 2))

        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (2400, 2800), cv2.INTER_LANCZOS4)
        paper = cv2.resize(paper, (2400, 2800), cv2.INTER_LANCZOS4)
        warped = cv2.resize(warped, (2400, 2800), cv2.INTER_LANCZOS4)
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        questionCnts = []
        answers = []
        # 对每一个轮廓进行循环处理
        for c in cnts:
            # 计算轮廓的边界框，然后利用边界框数据计算宽高比
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # 判断轮廓是否是答题框
            if w >= 40 and h >= 15 and ar >= 1 and ar <= 1.8:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                questionCnts.append(c)
                answers.append((cX, cY))
                cv2.circle(paper, (cX, cY), 7, (0, 0, 255), 3)

        self.processed_image = paper

    # 图像去噪 h=3:float, hColor=3:float, templateWindowSize=7, searchWindowSize=21 10, 10, 7, 21
    def image_denoising(self, h: float, hColor: float, templateWindowSize: int, searchWindowSize: int) -> None:
        self.processed_image = cv2.fastNlMeansDenoisingColored(self.original_image, None, h, hColor, templateWindowSize,
                                                               searchWindowSize)

    # 图像修复
    def image_restoration(self, mask):
        print(self.original_image.shape[0:2])
        mask = cv2.resize(mask, (self.original_image.shape[1], self.original_image.shape[0]))

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        self.processed_image = gray
        print(gray.shape)
        self.processed_image = cv2.inpaint(self.original_image, gray, 5, cv2.INPAINT_TELEA)
