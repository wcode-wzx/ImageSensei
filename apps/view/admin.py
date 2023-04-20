from flask import Blueprint, render_template
from apps.config import Explanation

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

explanation = Explanation()
@admin_bp.route('/index', methods=['GET','POST'])
def index():
    return render_template("base.html")

@admin_bp.route('/test', methods=['GET','POST'])
def test():
    return render_template("test.html")

@admin_bp.route('/show', methods=['GET','POST'])
def show():
    return render_template("show.html")

@admin_bp.route('/enhance_contrast', methods=['GET','POST'])
def enhance_contrast():
    return render_template("form/enhance_contrast.html", explanation=explanation.enhance_contrast())

@admin_bp.route('/geometric_transformation', methods=['GET','POST'])
def geometric_transformation():
    return render_template("form/geometric_transformation.html", explanation=explanation.geometric_transformation())

@admin_bp.route('/image_threshold', methods=['GET','POST'])
def image_threshold():
    return render_template("form/image_threshold.html", explanation=explanation.image_threshold())

@admin_bp.route('/smooth_image', methods=['GET','POST'])
def smooth_image():
    return render_template("form/smooth_image.html", explanation=explanation.smooth_image())

@admin_bp.route('/morphological_transformation', methods=['GET','POST'])
def morphological_transformation():
    return render_template("form/morphological_transformation.html", explanation=explanation.morphological_transformation())

@admin_bp.route('/edge_detection', methods=['GET','POST'])
def edge_detection():
    return render_template("form/edge_detection.html", explanation=explanation.edge_detection())

@admin_bp.route('/template_matching', methods=['GET','POST'])
def template_matching():
    return render_template("form/template_matching.html", explanation=explanation.template_matching())

@admin_bp.route('/hough_transform', methods=['GET','POST'])
def hough_transform():
    return render_template("form/hough_transform.html", explanation=explanation.hough_transform())

@admin_bp.route('/face_recognition', methods=['GET','POST'])
def face_recognition():
    return render_template("form/face_recognition.html", explanation=explanation.face_recognition())

@admin_bp.route('/answer_sheet_identification', methods=['GET','POST'])
def answer_sheet_identification():
    return render_template("form/answer_sheet_identification.html", explanation=explanation.answer_sheet_identification())

@admin_bp.route('/image_denoising', methods=['GET','POST'])
def image_denoising():
    return render_template("form/image_denoising.html", explanation=explanation.image_denoising())

@admin_bp.route('/image_restoration', methods=['GET','POST'])
def image_restoration():
    return render_template("form/image_restoration.html", explanation=explanation.image_restoration())