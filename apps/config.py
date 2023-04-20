class Explanation:

    # 答题卡识别
    def answer_sheet_identification(self):
        return """hshshh"""

    # 边缘检测
    def edge_detection(self):
        pass

    # 图像对比度增强
    def enhance_contrast(self):
        pass

    # 人脸识别
    def face_recognition(self):
        pass

    # 图像几何变换
    def geometric_transformation(self):
        return """
        <strong>缩放：</strong><br>缩放是调整图片的大小。 OpenCV 使用 cv.resize() 函数进行调整。可以手动指定图像的大小，也可以指定比例因子。<br>
        <strong>平移：</strong><br>平移变换是物体位置的移动。如果知道 （x，y） 方向的偏移量，假设为 (t_x,t_y)**，则可以创建如下转换矩阵 **M<br>
        $$M = \\begin{bmatrix}1&0&t_{x} , 0&1&t_{y} \end{bmatrix}$$<br>
        <strong>旋转：</strong><br>
        以 图片角度旋转图片的转换矩阵形式为：
        为了找到这个转换矩阵，opencv 提供了一个函数， cv.getRotationMatrix2D 。它将图像相对于中心旋转 90 度，而不进行任何缩放。<br>
        <strong>仿射变换：</strong><br>
        在仿射变换中，原始图像中的所有平行线在输出图像中仍然是平行的。为了找到变换矩阵，我们需要从输入图像中取三个点及其在输出图像中的对应位置。然后 cv.getAffineTransform 将创建一个 2x3 矩阵，该矩阵将传递给 cv.warpAffine 。
        """

    # 霍夫变换
    def hough_transform(self):
        pass

    # 图像去噪
    def image_denoising(self):
        pass

    # 图像修复
    def image_restoration(self):
        pass

    # 图像阈值
    def image_threshold(self):
        pass

    # 形态变换
    def morphological_transformation(self):
        pass

    # 平滑图像
    def smooth_image(self):
        pass

    # 模板匹配
    def template_matching(self):
        pass
