# -*- coding: utf-8 -*-
import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify, Blueprint, render_template
from apps.utils.ImgProcess import ImageProcessor

img_bp = Blueprint('img', __name__, url_prefix='/img')

imp = ImageProcessor()


# 接口：设置输入图像
@img_bp.route('/set_original_image', methods=['POST'])
def set_image():
    if request.method == 'POST':
        # 从请求中获取图片数据
        img_data = request.files.get('image').read()

        # 将图片数据转换为OpenCV格式
        nparr = np.fromstring(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        imp.set_original_image(img)
        imp.set_processed_image()
        return render_template('base.html')


@img_bp.route('/get_hwc', methods=['GET', 'POST'])
def get_hwc():
    if request.method == 'GET':
        res = []
        for key, im in {"original": imp.get_original_image(), "processed": imp.get_processed_image()}.items():
            try:
                if len(im.shape) == 3:
                    height, width, channels = im.shape
                    size = im.size
                else:
                    height, width = im.shape
                    channels = 1
                    size = im.size
                res.append({"type": key, "height": height, "width": width, "channels": channels, "size": size})
            except:
                res.append({"type": '...', "height": '...', "width": '...', "channels": '...', "size": '...'})
        return res


@img_bp.route('/get_original_image', methods=['GET', 'POST'])
def get_original_image():
    img_str = cv2.imencode('.jpg', imp.get_original_image())[1].tostring()
    return img_str, 200, {"Content-Type": "image/jpeg"}


@img_bp.route('/get_processed_image', methods=['GET', 'POST'])
def get_processed_image():
    img_str = cv2.imencode('.jpg', imp.get_processed_image())[1].tostring()
    return img_str, 200, {"Content-Type": "image/jpeg"}


@img_bp.route('/overwrite_original_image', methods=['GET'])
def overwrite_original_image():
    imp.overwrite_original_image()
    return jsonify({'status': 'success'})


# 对比度增强
@img_bp.route('/enhance_contrast', methods=['POST'])
def enhance_contrast():
    alpha = float(request.form.get('alpha'))
    beta = float(request.form.get('beta'))
    imp.enhance_contrast(alpha=alpha, beta=beta)
    return jsonify({'status': 'success'})


@img_bp.route('/geometric_transformation', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        operation = request.form.get('operation')
        if operation == 'scale':
            scale_factor = float(request.form.get('scale_factor'))
            # 处理缩放操作
            imp.scale(scale_factor=scale_factor)
            # return f"缩放因子：{scale_factor}"
        elif operation == 'translate':
            x = float(request.form.get('x'))
            y = float(request.form.get('y'))
            # 处理平移操作
            imp.translate(x=x, y=y)
            # return f"X：{x}，Y：{y}"
        elif operation == 'rotate':
            angle = float(request.form.get('angle'))
            # 处理旋转操作
            imp.rotate(angle=angle)
            # return f"角度：{angle}"
        elif operation == 'affine':
            src_points = request.form.get('src_points')
            dst_points = request.form.get('dst_points')
            # 处理仿射操作
            imp.affine(src_points=src_points, dst_points=dst_points)
            # return f"源点：{src_points}，目标点：{dst_points}"
        return jsonify({'status': 'success'})

# 图像阈值
@img_bp.route('/image_threshold', methods=['GET', 'POST'])
def image_threshold():
    if request.method == 'POST':
        function_name = request.form['select']
        if function_name == 'threshold':
            threshold_value = request.form['threshold-value']
            max_value = request.form['max-value']
            threshold_type = request.form['threshold-type']
            imp.threshold(threshold_value=threshold_value, max_value=max_value, threshold_type=threshold_type)
            print('Threshold Function: threshold_value={}, max_value={}, threshold_type={}'.format(threshold_value,
                                                                                                   max_value,
                                                                                                   threshold_type))
        elif function_name == 'adaptive_threshold':
            block_size = request.form['block-size']
            c = request.form['c']
            threshold_type = request.form['threshold-type']
            imp.adaptive_threshold(block_size=block_size, c=c, threshold_type=threshold_type)
            print('Adaptive Threshold Function: block_size={}, c={}, threshold_type={}'.format(block_size, c,
                                                                                               threshold_type))
        elif function_name == 'binary':
            print('Binary Function')
            imp.binary()
        elif function_name == 'grayscale':
            imp.grayscale()
            print('Grayscale Function')
        else:
            print('Invalid Function')

        return jsonify({'status': 'success Form Processed Successfully'})
    
# 平滑图像
@img_bp.route('/smooth_image', methods=['GET', 'POST'])
def smooth_image():
    if request.method == 'POST':
        # 获取请求中的滤波器类型和核大小
        filter_type = request.form['filter_type']
        kernel_size = int(request.form['kernel_size'])
        imp.set_kernel_size(kernel_size=kernel_size)

        if filter_type == 'mean':
            imp.mean_blur()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'gaussian':
            imp.gaussian_blur()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'median':
            imp.median_blur()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'bilateral':
            imp.bilateral_filter()
            return jsonify({'status': 'success Form Processed Successfully'})
        else:
            return jsonify({'error': 'Invalid filter type'})

# 形态转换
@img_bp.route('/morphological_transformation', methods=['GET', 'POST'])
def morphological_transformation():
    if request.method == 'POST':
        # 获取请求中的滤波器类型和核大小
        filter_type = request.form['filter_type']
        kernel_size = int(request.form['kernel_size'])
        imp.set_kernel(kernel_size=kernel_size)

        if filter_type == 'erode':
            imp.erode()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'dilate':
            imp.dilate()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'opening':
            imp.opening()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'closing':
            imp.closing()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'gradient':
            imp.gradient()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'tophat':
            imp.tophat()
            return jsonify({'status': 'success Form Processed Successfully'})
        elif filter_type == 'blackhat':
            imp.blackhat()
            return jsonify({'status': 'success Form Processed Successfully'})
        else:
            return jsonify({'error': 'Invalid filter type'})    



# 直方图均衡化
@img_bp.route('/equalize_histogram', methods=['POST'])
def equalize_histogram():
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(img)
    cv2.imwrite('processed_image.jpg', equalized_image)
    return jsonify({'status': 'success'})


# 图像滤波
@img_bp.route('/apply_filter', methods=['POST'])
def apply_filter():
    filter_type = request.form.get('filter_type')
    kernel_size = int(request.form.get('kernel_size'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_COLOR)
    if filter_type == 'blur':
        filtered_image = cv2.blur(img, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        filtered_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        filtered_image = cv2.medianBlur(img, kernel_size)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid filter type'})
    cv2.imwrite('processed_image.jpg', filtered_image)
    return jsonify({'status': 'success'})


# 图像锐化
@img_bp.route('/sharpen', methods=['POST'])
def sharpen():
    img = cv2.imread('input_image.jpg', cv2.IMREAD_UNCHANGED)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(img, -1, sharpen_kernel)
    cv2.imwrite('processed_image.jpg', sharpened_image)
    return jsonify({'status': 'success'})


# 形态学处理
@img_bp.route('/morphological_transform', methods=['POST'])
def morphological_transform():
    transform_type = request.form.get('transform_type')
    kernel_size = int(request.form.get('kernel_size'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if transform_type == 'erosion':
        transformed_image = cv2.erode(img, kernel, iterations=1)
    elif transform_type == 'dilation':
        transformed_image = cv2.dilate(img, kernel, iterations=1)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid transform type'})
    cv2.imwrite('processed_image.jpg', transformed_image)
    return jsonify({'status': 'success'})


# 边缘检测
@img_bp.route('/detect_edges', methods=['POST'])
def detect_edges():
    threshold1 = int(request.form.get('threshold1'))
    threshold2 = int(request.form.get('threshold2'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    edges_image = cv2.Canny(img, threshold1, threshold2)
    cv2.imwrite('processed_image.jpg', edges_image)
    return jsonify({'status': 'success'})


# 二值化
@img_bp.route('/threshold', methods=['POST'])
def threshold():
    threshold_value = int(request.form.get('threshold_value'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresholded_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite('processed_image.jpg', thresholded_image)
    return jsonify({'status': 'success'})
