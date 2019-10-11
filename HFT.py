import pandas as pd
import matplotlib.pyplot as plt
import numba

data = pd.read_csv( 'data\\tick-levelII\\bu1912 (8).csv', header=None)

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

class PosNode:
    def __init__(self, vol, price):
        self._vol = vol
        self._pos = price
    @property
    def vol(self):
        return self._vol
    @property
    def pos(self):
        return self._price

pnl_list = []
OPEN_SIGNAL = 2000
earning_target = -1
pnl = 0
is_holding = False

for i in range(len(fix_data) - 1):
    now_ask = fix_data.iloc[i, 3]
    now_bid = fix_data.iloc[i, 9]
    if is_holding:
        dir = pos.vol
        if dir == 1:
            if now_ask > earning_target:
                pnl += 8
                is_holding = False
            if now_ask < stoping_target:
                pnl -= 4
                is_holding = False
        if dir == -1:
            if now_bid < earning_target:
                pnl += 8
                is_holding = False
            if now_bid > stoping_target:
                pnl -= 4
                is_holding = False

    if not is_holding:
        if fix_data.iloc[i].loc['sell_rolling'] < -2000: # 假设下个tick以对手价买入
            this_cost = fix_data.iloc[i+1, 3]
            pos = PosNode(1, this_cost)
            earning_target = 2 * 4 + this_cost
            stoping_target = this_cost - 2 * 2
            is_holding = True

        if fix_data.iloc[i].loc['bid_rolling'] < -2000:  # 空头
            this_cost = fix_data.iloc[i + 1, 9]
            pos = PosNode(-1, this_cost)
            earning_target = this_cost - 2 * 4
            stoping_target = this_cost + 2 * 2
            is_holding = True

    pnl_list.append(pnl)


plt.plot(pnl_list)
plt.show()
data.iloc[:,1].plot()
plt.show()