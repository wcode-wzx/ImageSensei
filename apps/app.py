from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# 对比度增强
@app.route('/enhance_contrast', methods=['POST'])
def enhance_contrast():
    alpha = float(request.form.get('alpha'))
    beta = float(request.form.get('beta'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_UNCHANGED)
    enhanced_image = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    cv2.imwrite('processed_image.jpg', enhanced_image)
    return jsonify({'status': 'success'})

# 直方图均衡化
@app.route('/equalize_histogram', methods=['POST'])
def equalize_histogram():
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(img)
    cv2.imwrite('processed_image.jpg', equalized_image)
    return jsonify({'status': 'success'})

# 图像滤波
@app.route('/apply_filter', methods=['POST'])
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
@app.route('/sharpen', methods=['POST'])
def sharpen():
    img = cv2.imread('input_image.jpg', cv2.IMREAD_UNCHANGED)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(img, -1, sharpen_kernel)
    cv2.imwrite('processed_image.jpg', sharpened_image)
    return jsonify({'status': 'success'})

# 形态学处理
@app.route('/morphological_transform', methods=['POST'])
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
@app.route('/detect_edges', methods=['POST'])
def detect_edges():
    threshold1 = int(request.form.get('threshold1'))
    threshold2 = int(request.form.get('threshold2'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    edges_image = cv2.Canny(img, threshold1, threshold2)
    cv2.imwrite('processed_image.jpg', edges_image)
    return jsonify({'status': 'success'})

# 二值化
@app.route('/threshold', methods=['POST'])
def threshold():
    threshold_value = int(request.form.get('threshold_value'))
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresholded_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite('processed_image.jpg', thresholded_image)
    return jsonify({'status': 'success'})