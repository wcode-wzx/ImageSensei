import os
from dotenv import load_dotenv, find_dotenv

class EnvManager:
    def __init__(self):
        self.load_env()

    def load_env(self):
        """
        查找.env文件并加载环境变量
        """
        env_file = find_dotenv()
        load_dotenv(env_file, verbose=True)

    def get(self, key, default=None):
        """
        获取环境变量的值
        :param key:
        :param default:
        :return:
        """
        return os.getenv(key, default)

    def set(self, key, value):
        """
        设置环境变量的值
        :param key:
        :param value:
        :return:
        """
        os.environ[key] = value

    def get_env(self):
        """获取当前的环境"""
        return self.get('ENV', 'dev')

    def is_production(self):
        """判断是否生产环境"""
        return self.get_env() == 'prod'

    def is_development(self):
        """判断是否开发环境"""
        return self.get_env() == 'dev'

    def is_testing(self):
        """判断是否测试环境"""
        return self.get_env() == 'test'

    def get_test_mysql_cfg(self):
        """
        获取TEST数据配置信息
        :return:user, password, host, port, database
        """
        user = self.get("MYSQL_TEST_USER")
        password = self.get("MYSQL_TEST_PASSWORD")
        host = self.get("MYSQL_TEST_HOST")
        port = self.get("MYSQL_TEST_PORT")
        database = self.get("MYSQL_TEST_DATABASE")

        return user, password, host, port, database

    def get_formal_mysql_cfg(self):
        """
        获取FORMAL数据配置信息
        :return:user, password, host, port, database
        """
        user = self.get("MYSQL_FORMAL_USER")
        password = self.get("MYSQL_FORMAL_PASSWORD")
        host = self.get("MYSQL_FORMAL_HOST")
        port = self.get("MYSQL_FORMAL_PORT")
        database = self.get("MYSQL_FORMAL_DATABASE")

        return user, password, host, port, database