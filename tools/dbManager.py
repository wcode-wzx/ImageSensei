#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dbManager.py
@Time    :   2023/2/23 18:26
@Author  :   thyme
@Desc    :
'''
import time
import pandas as pd
from datetime import timedelta
from tools.envManager import EnvManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DataEngine():
    def __init__(self, mode):
        """
        选择数据库账户
        :param mode:test && formal
        """
        em = EnvManager()
        if mode == "test":
            user, password, host, port, database = em.get_test_mysql_cfg()
            self.database = database
            self.dbConnect(user, password, host, port, database)
        elif mode == "formal":
            user, password, host, port, database = em.get_formal_mysql_cfg()
            self.dbConnect(user, password, host, port, database)
            self.database = database
        else:
            raise ValueError("Invalid mode: {}".format(mode))


    def dbConnect(self, user, password, host, port, database):
        self.conn = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4')

    def dbExecute(self):
        """
        用于执行sql
        :return:
        """
        return self.conn

    # 获取运行时间
    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def insert_data(self, df, datatable, if_exists='append'):
        """
        把df插入数据到数据库
        :param df:
        :param datatable:
        :param if_exists:
        :return:
        """
        start_time = time.time()
        df.to_sql(datatable, self.conn, schema=self.database, if_exists=if_exists, index=False)
        print(f'插入数据成功需要的时间为{str(self.get_time_dif(start_time))}秒')

    def read_data(self, sql):
        """
        读取数据到df
        :param sql:aql语句
        :return:
        """
        start_time = time.time()
        pd_sql_data = pd.read_sql(sql, con=self.conn)
        print(f'查询数据成功需要的时间为{str(self.get_time_dif(start_time))}秒')
        return pd_sql_data

    def rd(self, sql):
        """
        读取数据到df
        :param sql:aql语句
        :return:
        """
        pd_sql_data = pd.read_sql(sql, con=self.conn)
        return pd_sql_data


class Database(DataEngine):
    """
    SQLAlchemy database helper class.
    """
    def __init__(self, mode):
        super().__init__(mode)
        self.engine = self.dbExecute()
        self.Session = sessionmaker(bind=self.engine)

    def create_all(self):
        """
        Create database tables.
        """
        Base.metadata.create_all(self.engine)

    def drop_all(self):
        """
        Drop all database tables.
        """
        Base.metadata.drop_all(self.engine)

    def add(self, instance):
        """
        Add an instance to the database.
        """
        session = self.Session()
        session.add(instance)
        session.commit()
        session.close()

    def delete(self, instance):
        """
        Delete an instance from the database.
        """
        session = self.Session()
        session.delete(instance)
        session.commit()
        session.close()

    def update(self, instance):
        """
        Update an instance in the database.
        """
        session = self.Session()
        session.merge(instance)
        session.commit()
        session.close()

    def query(self, cls):
        """
        Query all instances of a certain class.
        """
        session = self.Session()
        result = session.query(cls).all()
        session.close()
        return result


if __name__ == "__main__":

    DE = DataEngine(mode="test")
    df = DE.read_data("select * from test")
    print(df)
    # df["runoob_id"] = [i for i in range(4,4+len(df))]
    # DE.insert_data(df, "test")
    DE = DataEngine(mode="formal")
    res = DE.read_data('select * from sv_user limit 1')
    print(type(res), res)
