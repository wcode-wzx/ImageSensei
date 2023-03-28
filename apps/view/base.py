from flask import Flask, request, jsonify, Blueprint, render_template

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/index', methods=['GET','POST'])
def index():
    return render_template("base.html")

@admin_bp.route('/test', methods=['GET','POST'])
def test():
    return render_template("test.html")