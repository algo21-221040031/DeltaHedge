from index import *


class PriceMonteCarlo(AutoCall):

    def __init__(self):
        super().__init__()

    def get_price_matrix(self, sigma: float) -> None:
        """

        :param sigma: float, 建仓的时候需要做一个新参数
        :return: None, 对类中的参数做修改
        """
        rand_matrix = np.random.standard_normal(
            (self.num_trade_days, self.num_simu))
        # 随机微分项
        sde_rand_matrix = ((self.return_risk_free - self.return_dividend
                            - 0.5 * np.power(sigma, 2)) * self.time_delta
                           + sigma * np.sqrt(self.time_delta) * rand_matrix)
        exp_sde_rand_matrix = np.exp(sde_rand_matrix)
        # 涨跌停处理
        exp_sde_rand_matrix_up = np.where(
            exp_sde_rand_matrix > 1.1, 1.1, exp_sde_rand_matrix)
        exp_sde_rand_matrix_middle = np.where(
            exp_sde_rand_matrix_up < 0.9, 0.9, exp_sde_rand_matrix_up)
        cumprod_matrix = np.cumprod(exp_sde_rand_matrix_middle, axis=0)
        head = np.array([1] * self.num_simu)
        cumprod_matrix = np.vstack((head, cumprod_matrix))
        cumprod_dataframe = pd.DataFrame(cumprod_matrix)
        self.cum_return = cumprod_dataframe.copy(deep=True)
        self.cum_return_copy = cumprod_dataframe.copy(deep=True)

    def get_stock_path(self, init_price) -> None:
        """

        :param init_price: float, 初始价格
        :return: 类中参数的变动
        """
        self.stock_path = init_price * self.cum_return

    def get_expected_payoff(self, price_path: pd.DataFrame, knock_status: bool) -> dict:
        """
        规则设定：敲出>敲入>未敲入未敲出
        :param price_path: pd.DataFrame, 模拟的价格路径
        :param knock_status: bool, 是否敲入
        :return: dict, 预期收益
        """
        # 设置参数
        price_path = price_path
        risk_free_rate = self.return_risk_free  # 无风险利率
        num_trade_days = self.num_trade_days  # 产品存续期交易日天数
        time_delta = self.time_delta  # 时间间隔，反映在年化利率分散到每天
        return_coupon = self.return_bonus  # 红利票息
        return_ko = self.return_ko  # 敲出票息
        year = self.num_period  # 产品存续期
        principal = self.principal  # 产品名义本金
        num_simu = self.num_simu  # 模拟次数
        ki_bench = self.knock_in  # 敲入阈值
        ko_bench = self.knock_out  # 敲出阈值
        observe_point = self.observe_point  # 敲出观察日
        price_values = price_path.values  # 模拟价格路径的数值
        path_col = [i for i in range(num_simu)]
        observe_point_price = price_path.loc[observe_point].values
        # 计算现值
        if len(observe_point):
            # 判断是否发生过敲出
            # price_path 和 ko_bench 应该一起在求series_data时被处理
            ko_bench_matrix = np.tile(ko_bench, self.num_simu).reshape(
                (len(ko_bench), self.num_simu), order='F')
            ever_out = np.where(
                np.array(observe_point_price >= ko_bench_matrix).sum(axis=0) != 0)[0]
            path_ko_point = pd.DataFrame(price_path.loc[observe_point, ever_out])
            ko_bench_matrix = ko_bench_matrix[:, 0:len(ever_out)]
            path_ko_point = pd.DataFrame(path_ko_point > ko_bench_matrix)
            first_ko_date = path_ko_point.idxmax()
            # first_ko_data-price_path.index.min(): 折现的时候要考虑是第几天
            # 要把年化的收益率处理成实际天数
            ko_discount_factor = np.exp(
                -risk_free_rate * (first_ko_date - price_path.index.min()) * time_delta)  # risk_free_rate: 年化
            return_ko = pd.DataFrame(
                return_ko, index=observe_point, columns=['return_ko'])
            payoff_ever_ko = (1 + first_ko_date.apply(
                lambda x: return_ko.loc[x, 'return_ko'] * x) * time_delta) * principal * ko_discount_factor
            payoff_ever_ko = payoff_ever_ko.sum()
        else:
            ever_out = []
            payoff_ever_ko = 0
        ever_in = np.where(np.min(price_path, axis=0) <= ki_bench)[0]
        ever_io = list(set(ever_out) & set(ever_in))  # 既发生敲入有发生敲出
        # 判断是否发生过敲入
        # 如果一旦发生敲入，则之后不再有未敲入未敲出的情景
        if not knock_status:
            only_in = list(set(ever_in) - set(ever_io))
            payoff_only_in = np.exp(-risk_free_rate * len(price_path) * time_delta) * principal * price_values[
                -1, only_in].sum()
            not_out_in = list(set(path_col) ^ set(ever_out) ^ set(only_in))
            payoff_noi = len(not_out_in) * (1 + return_coupon) * principal * np.exp(
                -risk_free_rate * len(price_path) * time_delta)
            payoff_expect = (payoff_ever_ko + payoff_only_in +
                             payoff_noi) / self.num_simu
        else:
            only_in = list(set(path_col) - set(ever_out))
            payoff_only_in = np.exp(-risk_free_rate * len(price_path)
                                    * time_delta) * principal * price_values[-1, only_in].sum()
            payoff_noi = 0
            payoff_expect = (payoff_ever_ko + payoff_only_in + payoff_noi) / self.num_simu
        return {'expect_payoff': payoff_expect,
                'payoff_ever_ko': payoff_ever_ko / len(ever_out) if len(ever_out) else payoff_ever_ko,
                'payoff_only_in': payoff_only_in / len(only_in) if len(only_in) else only_in,
                'payoff_noi': payoff_noi / len(ever_io) if len(ever_io) else payoff_noi}

    def get_autocall_price(self, knock_status) -> dict:
        """
        
        :param knock_status: 是否已敲入
        :return: dict, 返回雪球估值
        """
        price_path = self.stock_path
        autocall_price_result = self.get_expected_payoff(price_path, knock_status)
        autocall_price = autocall_price_result['expect_payoff']
        autocall_price_ko_part = autocall_price_result['payoff_ever_ko']
        autocall_price_ki_part = autocall_price_result['payoff_only_in']
        autocall_price_noi_part = autocall_price_result['payoff_noi']
        return {'autocall_price': autocall_price,
                'autocall_price_ko_part': autocall_price_ko_part,
                'autocall_price_ki_part': autocall_price_ki_part,
                'autocall_price_noi_part': autocall_price_noi_part}

    def get_delta(self, index_price: float, knock_status: int, price_delta: float) -> dict:
        """

        :param index_price: 指数价格，需要和init_price比较，处理成涨跌幅的形式
        :param price_delta: 指数价格变动幅度
        :param knock_status: 是否敲入
        :return: delta
        """
        up_price = index_price + price_delta
        self.get_stock_path(up_price)
        price_up_price = self.get_autocall_price(knock_status)[
            'autocall_price']
        down_price = index_price - price_delta
        self.get_stock_path(down_price)
        price_down_price = self.get_autocall_price(knock_status)[
            'autocall_price']
        calc_delta = (price_up_price - price_down_price) / (2 * price_delta)
        delta = calc_delta / self.principal if (calc_delta / self.principal) < 2 else 2
        return {'delta': delta, 'up_price': up_price, 'down_price': down_price, 'price_up_price': price_up_price,
                'price_down_price': price_down_price}

    def get_gamma(self, index_price: float, knock_status: int, price_delta: float) -> dict:
        """

        :param index_price: 指数价格，需要和init_price比较，处理成百分比的形式
        :param knock_status: 是否敲入
        :param price_delta: 指数价格变动幅度
        :return: gamma
        """
        self.get_stock_path(index_price)
        price_price = self.get_autocall_price(knock_status)['expected_payoff']
        up_price = index_price + price_delta
        self.get_stock_path(up_price)
        price_up_price = self.get_autocall_price(knock_status)[
            'autocall_price']
        down_price = index_price - price_delta
        self.get_stock_path(down_price)
        price_down_price = self.get_autocall_price(knock_status)[
            'autocall_price']
        gamma = (price_up_price - price_down_price +
                 2 * price_price) / (price_delta ** 2)
        return {'gamma': gamma, 'price': index_price, 'up_price': up_price, 'down_price': down_price,
                'price_price': price_price, 'price_up_price': price_up_price, 'price_down_price': price_down_price}

    def get_series_delta(self, price_series: list, price_delta: float) -> pd.DataFrame:
        """

        :param price_series: list, 挂钩标的价格序列
        :param price_delta: float, 价格变动幅度
        :return: pd.DataFrame, 定价|Delta|是否敲入
        """
        price_array = np.zeros(self.num_trade_days + 1)
        delta_array = np.zeros(self.num_trade_days + 1)
        knock_status_array = np.zeros(self.num_trade_days + 1)
        ob_point = self.observe_copy
        if len(self.observe_point) - len(ob_point):
            self.observe_point.clear()
            self.observe_point.append(self.observe_copy)
        for i in range(len(price_series)-1):
            spot_price = price_series[i] / price_series[0]
            knock_status = 1 if (spot_price < self.knock_in) or knock_status_array[:i].sum() else 0
            knock_status_array[i] = float(knock_status)
            np.random.seed(10)
            self.cum_return = self.cum_return_copy.copy(deep=True)
            self.cum_return = self.cum_return.iloc[i:, :]
            delta = self.get_delta(spot_price, knock_status, price_delta)['delta']
            price_result = self.get_autocall_price(knock_status)
            if len(self.observe_point):
                if i == self.observe_point[0] and spot_price < self.knock_out[0]:
                    self.observe_point.pop(0)
                    self.knock_out.pop(0)
                    self.return_ko.pop(0)
                    # self.observe_point = self.observe_point[i:]
            # 使用[i:,:]确保是dataframe
            if i == self.observe_point[0] and spot_price > self.knock_out[0]:
                knock_status_array[i] = -1
                break
            price_array[i] = price_result['autocall_price']
            delta_array[i] = delta
            print(f"第{i}次估值：")
            print('敲出部分价值：', price_result['autocall_price_ko_part'])
            print('敲入部分价值：', price_result['autocall_price_ki_part'])
            print('未敲入敲出部分价值：', price_result['autocall_price_noi_part'])
            print(self.knock_out)
            print(self.observe_point)
            print(self.cum_return)
            print(self.return_ko)
            print("票息列表的长度为：", len(self.return_ko))
        delta_array = delta_array
        price_array = price_array / self.principal
        output = {'delta': delta_array,
                  'price': price_array,
                  'knock': knock_status_array}
        result_output = pd.DataFrame(output)
        # self.cum_return.to_excel("final_result.xlsx")
        # self.cum_return_copy.to_excel("init_result.xlsx")

        return result_output


if __name__ == "__main__":
    test = PriceMonteCarlo()
