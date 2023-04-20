class Explanation:

    # 答题卡识别
    def answer_sheet_identification(self):
        return """先灰度化再二值化。<br>
        边缘检测划出选择题区域，如果直接进行边缘检测的话，干扰特别多，所以这个时候需要采取腐蚀膨胀等一系列处理。<br>
        使用cv2.findContours()函数来查找涂黑区域的轮廓。<br>
        最后定位做标记。
        """

    # 边缘检测
    def edge_detection(self):
        pass

    # 图像对比度增强
    def enhance_contrast(self):
        pass

    # 人脸识别
    def face_recognition(self):
        return """
        OpenCV 已经包含许多面部，眼睛，微笑等预先训练的分类器。这些 XML 文件存储在 opencv/data/haarcascades/文件夹 中。<br>
        我们在图像中找到面孔。如果找到了面，它会将检测到的面的位置返回为 Rect（x，y，w，h）。一旦我们获得这些位置，我们就可以为脸部创建感兴趣区域。<br>
        包含两个参数：<br>
        scaleFactor “指定每次图像缩小的比例” <br>
        minNeighbors 指定每个候选矩形有多少个“邻居”
        """

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
