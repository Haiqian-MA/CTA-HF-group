import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numba
'''
仓位：有多少开多少：固定；或者看对手量

现象（基于已经做出来的）：
（1）波动率：波动率高效果好（短期波动率有好的方法测，比如用原油来预测）
（2）每天开盘交易一段时间就能判断全天的情况，统计可以加个阈值规避掉今天交易
（3）基于五档行情可以测度多大量可以撬动，产生大的上涨，（一天沥青90多笔，白银200多笔）
（4）反转趋势分开做

止盈止损：动态止损，最后止损出
'''

data = pd.read_csv('data/tick-levelII/bu1912 (6).csv', header=None)

data = data.iloc[:,[3,4,5,10,15,16,17,18,19,20,25,26,27,28,29]]

data.iloc[:, 2] = data.iloc[:, 2].diff()
data['bid_change'] = None
data['ask_change'] = None
#@numba.jit(nopython=True)


def calc_change_bid(bid_1, bid_size_1, bid_2, bid_size_2, tick_size=2):
    if bid_1 == bid_2:  ## no change in shape
        return bid_size_2[0] - bid_size_1[0]
    ans = 0
    if bid_1 < bid_2:   ## 向上击穿
        for i in range(5):
            if bid_1 + i * tick_size < bid_2:
                ans += bid_size_2[i]
            if bid_1 + i * tick_size == bid_2:
                ans += max(bid_size_2[i] - bid_size_1[0], 0)
        return ans

    if bid_1 > bid_2:   ## 向下击穿
        for j in range(5):
            if bid_1 - j * tick_size > bid_2:
                ans -= bid_size_1[j]
            if bid_1 - j * tick_size == bid_2:
                ans += min(bid_size_2[0] - bid_size_1[j], 0)
        return ans


def calc_change_ask(ask_1, ask_size_1, ask_2, ask_size_2, tick_size=2):
    if ask_1 == ask_2:  ## no change in shape
        return ask_size_2[0] - ask_size_1[0]
    ans = 0
    if ask_1 > ask_2:
        for i in range(5):
            if ask_1 - i * tick_size > ask_2:
                ans += ask_size_2[i]
            if ask_1 - i * tick_size == ask_2:
                ans += max(ask_size_2[i] - ask_size_1[0], 0)
        return ans

    if ask_1 < ask_2:
        for j in range(5):
            if ask_1 + j * tick_size < ask_2:
                ans -= ask_size_1[j]
            if ask_1 + j * tick_size == ask_2:
                ans -= max(ask_size_1[0] - ask_size_2[j], 0)
        return ans


bid_change = []
for i in range(1, len(data)):
    bid_change.append( calc_change_bid(data.iloc[i-1, 9], data.iloc[i-1, 10: 15].values,
                                 data.iloc[i, 9], data.iloc[i, 10: 15].values) )


ask_change = []
for i in range(1, len(data)):
    ask_change.append( calc_change_ask(data.iloc[i-1, 3], data.iloc[i-1, 4: 9].values,
                                 data.iloc[i, 3], data.iloc[i, 4: 9].values) )

data = data.iloc[1:,:]

data['bid_change'] = bid_change
data['ask_change'] = ask_change
ROLLING_ = 180

data['bid_rolling'] = data['bid_change'].rolling(ROLLING_).sum()
data['sell_rolling'] = data['ask_change'].rolling(ROLLING_).sum()

fix_data = data.dropna()
#%%
vol_all = np.full([data.shape[0], 6], np.nan)
vol_all[:, 2] = 0
vol_all[:, 5] = 0

def judge_price():
    pass

def send_order(vol_all, tick, vol_num, price, is_open):

    if not is_open:
        vol_all[tick:, 5] += (price-vol_all[tick, 3]) * vol_all[tick, 4]

    vol_all[tick, 0] = vol_num
    vol_all[tick, 1] = price
    vol_all[tick:, 2] = is_open
    vol_all[tick:, 3] = price
    vol_all[tick:, 4] = vol_num

for i in range(data.shape[0]):
    if i < 1000:
        continue

    last_price_array = data.iloc[i-500:i, 1].values

    is_open = vol_all[i, 2]
    bid_rolling = data.iloc[i, :]['bid_rolling']
    sell_rolling = data.iloc[i, :]['sell_rolling']

    if not is_open:
        if (last_price_array[-1] >= np.max(last_price_array[:-1]) and
            last_price_array[-1] > np.min(last_price_array[:-1]+20)):
            if bid_rolling > 2000 and sell_rolling < -2000:
                price = data.iloc[i + 1, 9] #买一价卖出
                send_order(vol_all, i, -1, price, 1)
                print(i, '卖开')

        if (last_price_array[-1] <= np.min(last_price_array[:-1]) and
            last_price_array[-1] < np.max(last_price_array[:-1]-20)):
            if sell_rolling > 2000 and bid_rolling < -2000:
                price = data.iloc[i + 1, 3] #卖一价买入
                send_order(vol_all, i, 1, price, 1)
                print(i, '买开')
    else:
        open_price = vol_all[i, 3]
        vol = vol_all[i, 4]
        if vol > 0:#买入持仓
            if last_price_array[-1]-open_price >= 10:#止盈
                price = data.iloc[i + 1, 9]  # 买一价卖出
                send_order(vol_all, i, -1, price, 0)
                print(i, '买开止盈')
            if last_price_array[-1]-open_price <= -10:#止损
                price = data.iloc[i + 1, 9]  # 买一价卖出
                send_order(vol_all, i, -1, price, 0)
                print(i, '买开止损')
        elif vol < 0:#卖出持仓
            if last_price_array[-1]-open_price <= -10:#止盈
                price = data.iloc[i + 1, 3]  # 卖一价买入
                send_order(vol_all, i, 1, price, 0)
                print(i, '卖开止盈')
            if last_price_array[-1]-open_price >= 10:#止损
                price = data.iloc[i + 1, 3]  # 卖一价买入
                send_order(vol_all, i, 1, price, 0)
                print(i, '卖开止损')

#%%
plt.figure()
plt.plot(vol_all[:, 5])
plt.show()