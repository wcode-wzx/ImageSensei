
# encoding=utf-8
from apps import create_app
from tools import EnvManager

em = EnvManager()

app = create_app()


if __name__ == '__main__':
    if em.is_development():
        app.run(host=em.get("ENV_HOST"), port=em.get("ENV_PORT_DEV"), debug=True, threaded=True)
    elif em.is_production():
        app.run(host=em.get("ENV_HOST"), port=em.get("ENV_PORT_PROD"), debug=False, threaded=True)
    elif em.is_testing():
        app.run(host=em.get("ENV_HOST"), port=em.get("ENV_PORT_TEST"), debug=False, threaded=True)
    else:
        ValueError("Invalid ENV: {}".format(em.get("ENV")))

