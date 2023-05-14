import numpy as np
import pandas as pd

simu_para = dict()
simu_para['r_interest'] = 0.03  # 无风险利率
simu_para['r_dividend'] = 0.015 + 0.06  # 红利率

simu_para['n_period'] = 2  # 年份
simu_para['n_nature_days'] = 365 * 2
simu_para['n_trade_days'] = 240 * 2
simu_para['n_simu'] = 10000

basic_para = dict()
basic_para['knock_in'] = 0.75  # 敲入阈值
basic_para['knock_out'] = 1  # 敲出阈值
basic_para['return_ko'] = 0.15  # 敲出票息(年化)
basic_para['return_bonus'] = 0.15  # 红利票息(年化)
basic_para['num_tdays_month'] = 20  # 一个月20个交易日
basic_para['lock_period'] = 3  # 封闭观察期
basic_para['principal'] = 100000000  # 名义本金


# 为不同结构做准备，敲出阈值和敲出票息根据敲出观察日一起构建


class AutoCall:

    def __init__(self, monte_carlo_para=None, product_info=None):
        if monte_carlo_para is None:
            monte_carlo_para = simu_para
        self.mc_para = monte_carlo_para
        self.return_risk_free = self.mc_para['r_interest']  # 无风险利率
        self.return_dividend = self.mc_para['r_dividend']  # 红利率
        self.num_period = self.mc_para['n_period']  # 年数
        self.num_nature_days = self.mc_para['n_nature_days']  # 合约自然日天数
        self.num_trade_days = self.mc_para['n_trade_days']  # 合约交易日天数
        self.time_delta = self.num_period / self.num_nature_days
        # self.time_delta = self.num_period / self.num_trade_days
        self.num_simu = self.mc_para['n_simu']

        if product_info is None:
            product_info = basic_para
        self.product_para = product_info
        self.knock_in = self.product_para['knock_in']  # 敲出阈值
        self.return_bonus = self.product_para['return_bonus']  # 红利票息
        self.num_tdays = self.product_para['num_tdays_month']  # 每月的交易日
        self.lock_period = self.product_para['lock_period']  # 封闭观察期
        self.observe_point = [k for k in range(
            self.lock_period * self.num_tdays, self.num_trade_days + self.num_tdays, self.num_tdays)]
        self.calendar = [k for k in range(self.num_trade_days + 1)]
        # 标准款雪球 如果有其他结构 需要在类中重新定义
        self.return_ko = [self.product_para['return_ko']] * len(self.observe_point)
        self.knock_out = [self.product_para['knock_out']] * len(self.observe_point)
        self.principal = self.product_para['principal']

        self.stock_path = pd.DataFrame()  # 用于保存股票价格路径
        self.cum_return = pd.DataFrame()  # 用于保存累计收益率路径
        self.cum_return_copy = pd.DataFrame()  # 用于保存累计收益率路径
        self.simulation_matrix = np.zeros([self.num_trade_days, self.num_simu])  # 用于存储蒙卡路径
        self.return_lst = []  # 存储收益结果
        self.observe_copy = []
        self.observe_copy.extend(self.observe_point)

        # 写在DeltaHedge类里面
        # basic_para['buy_cost'] = 0.0005  # delta对冲买入现货手续费
        # basic_para['sell_cost'] = 0.0009  # delta对冲卖出现货手续费
        # basic_para['F'] = 10000  # 面值
        # basic_para['init_net_value'] = 100000000  # 计算对冲收益时初始账户净值
        # basic_para['capital_cost_rate'] = 0.05  # 资金使用成本--百分比
        # basic_para['trade_cost_rate'] = 0.00015  # 交易成本--绝对值
        # basic_para['path_count'] = 0
