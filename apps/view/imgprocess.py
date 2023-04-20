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
        return render_template('extend.html')


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

        # 边缘检测


@img_bp.route('/edge_detection', methods=['POST'])
def edge_detection():
    if request.method == 'POST':
        minVal = int(request.form.get('minVal'))
        maxVal = int(request.form.get('maxVal'))
        imp.edge_detection(minVal=minVal, maxVal=maxVal)
        return jsonify({'status': 'edge_detection success'})


# 模板匹配
@img_bp.route('/template_matching', methods=['POST'])
def template_matching():
    if request.method == 'POST':
        template_data = request.files.get('image').read()
        method = request.form['method']

        # 将图片数据转换为OpenCV格式
        nparr = np.fromstring(template_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        imp.template_matching(img, method)
        return jsonify({'status': 'template_matching success'})


# 霍夫变换
@img_bp.route('/hough_transform', methods=['POST'])
def hough_transform():
    if request.method == 'POST':
        operation = request.form.get('select')
        if operation == 'line':
            rho_res = request.form.get('rho_res')
            threshold = request.form.get('threshold')
            imp.detect_lines(rho_res=int(float(rho_res)), threshold=int(float(threshold)))
        elif operation == 'circles':
            param1 = request.form.get('param1')
            param2 = request.form.get('param2')
            minRadius = request.form.get('minRadius')
            maxRadius = request.form.get('maxRadius')
            imp.detect_circles(param1=float(param1), param2=float(param2), minRadius=int(minRadius),
                               maxRadius=int(maxRadius))
        else:
            pass
        return jsonify({'status': 'template_matching success'})


# 人脸识别
@img_bp.route('/face_recognition', methods=['POST'])
def face_recognition():
    if request.method == 'POST':
        scaleFactor = request.form.get('scaleFactor')
        minNeighbors = request.form.get('minNeighbors')
        if imp.face_recognition(float(scaleFactor), int(minNeighbors)):
            return jsonify({'code': 200})
        else:
            return jsonify({'code': 0})


# answer_sheet_identification 答题卡识别
@img_bp.route('/answer_sheet_identification', methods=['POST'])
def answer_sheet_identification():
    if request.method == 'POST':
        imp.answer_sheet_identification()
        return jsonify({'code': 200})


# 图像去噪
@img_bp.route('/image_denoising', methods=['POST'])
def image_denoising():
    if request.method == 'POST':
        h = request.form.get('h')
        hColor = request.form.get('hColor')
        templateWindowSize = request.form.get('templateWindowSize')
        searchWindowSize = request.form.get('searchWindowSize')
        imp.image_denoising(h=float(h), hColor=float(hColor), templateWindowSize=int(templateWindowSize),
                            searchWindowSize=int(searchWindowSize))
        return jsonify({'code': 200})


# 图像修复
@img_bp.route('/image_restoration', methods=['POST'])
def image_restoration():
    if request.method == 'POST':
        template_data = request.files.get('image').read()
        # 将图片数据转换为OpenCV格式
        nparr = np.fromstring(template_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        imp.image_restoration(img)
        return jsonify({'code': 200})
