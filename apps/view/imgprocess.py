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
        # for im in [imp.get_original_image(), imp.get_processed_image()]:
        for key, im in {"original":imp.get_original_image(), "processed":imp.get_processed_image()}.items():
            try:
                height, width, channels = im.shape
                size = im.size
                res.append({"type":key, "height": height, "width": width, "channels": channels, "size": size})
            except:
                res.append({"type":'...', "height": '...', "width": '...', "channels": '...', "size": '...'})
        print(res)
        return res

@img_bp.route('/get_original_image', methods=['GET', 'POST'])
def get_original_image():
    img_str = cv2.imencode('.jpg', imp.get_original_image())[1].tostring()
    return img_str, 200, {"Content-Type": "image/jpeg"}


@img_bp.route('/get_processed_image', methods=['GET', 'POST'])
def get_processed_image():
    img_str = cv2.imencode('.jpg', imp.get_processed_image())[1].tostring()
    return img_str, 200, {"Content-Type": "image/jpeg"}


# 对比度增强
@img_bp.route('/enhance_contrast', methods=['POST'])
def enhance_contrast():
    alpha = float(request.form.get('alpha'))
    beta = float(request.form.get('beta'))
    imp.enhance_contrast(alpha=alpha, beta=beta)
    return jsonify({'status': 'success'})


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
