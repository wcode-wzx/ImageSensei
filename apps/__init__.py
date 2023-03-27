# Flask
import os
from flask import Flask
from server.view.recommend import recommend_bp
from server.view.upload import upload_bp
from server.view.metrics import metrics_bp
from tools import EnvManager

em = EnvManager()


def init_view(app):
    app.register_blueprint(recommend_bp)
    app.register_blueprint(upload_bp)
    # 非生产环境，加载指标统计蓝图 metrics
    if not em.is_production():
        app.register_blueprint(metrics_bp)


def create_app(config_name=None):
    app = Flask(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), template_folder="./server/templates")
    # Define a flask app
    app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

    if not config_name:
        # 尝试从本地环境中读取
        config_name = os.getenv('FLASK_CONFIG', 'development')

    # 注册路由
    init_view(app)

    return app