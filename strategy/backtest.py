from price import *
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    index_data = pd.read_excel(
            "../data/000852_price_sigma.xlsx", sheet_name="required")
    # 每一个月建仓一次(按照交易日20天计算)
    date_list = index_data.date.tolist()
    
    open_a_position = []

    # for i in range(0, len(date_list), 20):
    x1 = time.time()
    i = 1000
    open_a_position.append(date_list[i])
    init_price = index_data.loc[i, 'price']
    sigma = index_data.loc[i, 'sigma'] / 100
    price_series = index_data.loc[i:i + simu_para['n_trade_days'], 'price'].reset_index(drop=True) / init_price
    # 一个合约是一个对象
    autocall = PriceMonteCarlo()
    autocall.get_price_matrix(sigma)
    autocall.get_stock_path(1)
    backtest_result = autocall.get_series_delta(price_series, 0.01)
    backtest_result['index_price'] = price_series
    x2 = time.time()

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    axes.plot(backtest_result.delta)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    axes.plot(backtest_result.price)








