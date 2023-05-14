# test.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Base(object):
    """
    Base: 产品信息
    """

    def __init__(self, base_info):
        """

        :param base_info: 产品信息
        """
        self.prod = base_info
        # self.n_days = 240 * 2  # 产品存续天数
        # self.period = 2  # 产品存续期
        # self.knock_out = 1.1  # 敲出阈值
        # self.knock_in = 0.75  # 敲入阈值
        # self.r_interest = 0.03  # 无风险利率
        # self.r_dividend = 0.015 + 0.06  # 红利率
        # self.r_ko = 0.2  # 敲出收益
        # self.r_bonus = 0.2  # 未敲入敲出收益
        # self.n_month = 20  # 每月天数
        # self.obs_freq = 5  # 观测频率
        self.n_days = self.prod['n_days']  # 产品存续天数
        self.period = self.prod['period']  # 产品存续期
        self.knock_out = self.prod['knock_out']  # 敲出阈值
        self.knock_in = self.prod['knock_in']  # 敲入阈值
        self.r_interest = self.prod['r_interest']  # 无风险利率
        self.r_dividend = self.prod['r_dividend']  # 红利率
        self.r_ko = self.prod['r_ko']  # 敲出收益
        self.r_bonus = self.prod['r_bonus']  # 未敲入敲出收益
        self.n_month = self.prod['n_month']  # 每月天数
        self.obs_freq = self.prod['obs_freq']  # 观测频率


class Simulation(Base):
    """
    Simulation: 蒙卡生成价格随机路径
    """

    def __init__(self, date, price, sigma):
        """

        :param date: 估值日
        :param price: 估值日收盘价
        :param sigma: 估值日向前滚动60日波动率
        """
        super(Simulation, self).__init__(base_info=product_info)
        self.n_simu = 10000
        self.dt = self.period / self.n_days
        self.init_price = price
        self.sigma = sigma
        self.sde_matrix = pd.DataFrame()
        self.price_path = pd.DataFrame()
        return

    def get_simulation_matrix(self):
        rand_matrix = np.random.standard_normal((self.n_days, self.n_simu))
        # 随机微分项
        sde_rand_matrix = (self.r_interest - self.r_dividend - 0.5 * np.power(self.sigma,
                                                                              2)) * self.dt + self.sigma * np.sqrt(
            self.dt) * rand_matrix
        exp_sde_rand_matrix = np.exp(sde_rand_matrix)
        # 涨跌停处理
        exp_sde_rand_matrix_up = np.where(exp_sde_rand_matrix > 1.1, 1.1, exp_sde_rand_matrix)
        exp_sde_rand_matrix_middle = np.where(exp_sde_rand_matrix_up < 0.9, 0.9, exp_sde_rand_matrix_up)
        exp_sde_rand_matrix_middle[0] = 1
        cumprod_matrix = np.cumprod(exp_sde_rand_matrix_middle, axis=0)
        cumprod_dataframe = pd.DataFrame(cumprod_matrix)
        self.sde_matrix = cumprod_dataframe.copy(deep=True)

        return cumprod_matrix

    def get_price_matrix(self):
        self.price_path = self.init_price * self.sde_matrix

        return self.price_path

    def plot_price_path(self):
        sns.set()
        plt.plot(self.price_path)


class SnowBall(Base):
    """
    明确敲入敲出规则
    计算雪球产品净值
    生成敲入敲出矩阵
    """

    def __init__(self):
        super(SnowBall, self).__init__(base_info=product_info)

    def get_expect_payoff(self):
        pass


if __name__ == "__main__":
    # 产品信息
    product_info = dict()
    product_info['period'] = 2
    product_info['n_days'] = 240 * product_info.get('period')
    product_info['knock_out'] = 1.1
    product_info['knock_in'] = 0.75
    product_info['r_interest'] = 0.03
    product_info['r_dividend'] = 0.015 + 0.06
    product_info['r_ko'] = 0.2
    product_info['r_bonus'] = 0.2
    product_info['n_month'] = 20
    product_info['obs_freq'] = 5
