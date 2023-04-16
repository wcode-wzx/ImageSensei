from flask import Blueprint, render_template

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

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
    return render_template("form/enhance_contrast.html")

@admin_bp.route('/geometric_transformation', methods=['GET','POST'])
def geometric_transformation():
    return render_template("form/geometric_transformation.html")

@admin_bp.route('/image_threshold', methods=['GET','POST'])
def image_threshold():
    return render_template("form/image_threshold.html")

@admin_bp.route('/smooth_image', methods=['GET','POST'])
def smooth_image():
    return render_template("form/smooth_image.html")

@admin_bp.route('/morphological_transformation', methods=['GET','POST'])
def morphological_transformation():
    return render_template("form/morphological_transformation.html")