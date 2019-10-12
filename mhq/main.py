#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

data = pd.read_csv('data/tick-levelII/bu1912 (6).csv', header=None)

#%%
plt.plot(data[4])
plt.show()

#%%
#思路：当价格达到一段时间内的最大值或者最小值，且做空、做空方，即卖方或买方前三档的挂单量明显大于对手方

#加减仓数手数，成交价格，开平仓，最新成交价, 总仓位，PNL
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

    last_price_array = data.loc[i-1000:i, 4].values
    last_sell_vol = data.loc[i-20:i+1, 15:18].values #取前三列
    last_buy_vol = data.loc[i-20:i+1, 25:28] .values # 取前三列
    is_open = vol_all[i, 2]
    his_sell = np.mean(last_sell_vol[:-5, :])
    now_sell = np.mean(last_sell_vol[-5:, :])
    his_buy = np.mean(last_buy_vol[:-5, :])
    now_buy = np.mean(last_buy_vol[-5:, :])

    if not is_open:
        if (last_price_array[-1] >= np.max(last_price_array[:-1]) and
            last_price_array[-1] > np.min(last_price_array[:-1]+20)):
            if now_sell > now_buy * 2:
                price = data.loc[i + 1, 20] #买一价卖出
                send_order(vol_all, i, -1, price, 1)
                print(i, '卖开')

        if (last_price_array[-1] <= np.min(last_price_array[:-1]) and
            last_price_array[-1] < np.max(last_price_array[:-1]-20)):
            if now_buy > now_sell * 2:
                price = data.loc[i + 1, 10] #卖一价买入
                send_order(vol_all, i, 1, price, 1)
                print(i, '买开')
    else:
        open_price = vol_all[i, 3]
        vol = vol_all[i, 4]
        if vol > 0:#买入持仓
            if last_price_array[-1]-open_price >= 10:#止盈
                price = data.loc[i + 1, 20]  # 买一价卖出
                send_order(vol_all, i, -1, price, 0)
                print(i, '买开止盈')
            if last_price_array[-1]-open_price <= -6:#止损
                price = data.loc[i + 1, 20]  # 买一价卖出
                send_order(vol_all, i, -1, price, 0)
                print(i, '买开止损')
        elif vol < 0:#卖出持仓
            if last_price_array[-1]-open_price <= -10:#止盈
                price = data.loc[i + 1, 10]  # 卖一价买入
                send_order(vol_all, i, 1, price, 0)
                print(i, '卖开止盈')
            if last_price_array[-1]-open_price >= 6:#止损
                price = data.loc[i + 1, 10]  # 卖一价买入
                send_order(vol_all, i, 1, price, 0)
                print(i, '卖开止损')

#%%
plt.figure()
plt.plot(vol_all[:, 5])
plt.show()


#%%
'''
def cc(num):
    return num*(num-1)*(num-2)/6

def unique(arr):
    all = []
    for a in arr:
        if a not in all:
            all.append(a)
    return all


def countTriplets(arr, n, r):
    cnt = 0
    if r == 1:
        item = unique(arr)
        print(item)
        for i in item:
            num = arr.count(i)
            cnt += cc(num)

    elif r != 1:
        for i in range(n):
            a = arr[i]
            for j in range(i+1,n):
                b = arr[j]
                if b != a*r:
                    continue
                for k in range(j+1,n):
                    c = arr[k]
                    if c != b*r:
                        continue
                    else:
                        print('cnt',a,b,c)
                        cnt += 1
    return cnt

#%%

# !/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'interpolate' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY instances
#  3. FLOAT_ARRAY price
#
def pop_laps(instances, price):
    false = []
    for i, p in enumerate(price):
        if p <= 0:
            false.append(i)

    for i in range(len(false)):
        f = false[len(false) - i - 1]
        instances.pop(f)
        price.pop(f)

def double_sort(instances, price):
    instances2 = instances.copy()
    instances.sort()
    p = []
    for ins in instances:
        p.append(price[instances2.index(ins)])
    price = p

def judge_index(instances, n):
    i = 0
    for i, instance in enumerate(instances):
        if n < instances[0]:
            return 0, -1
        if n == instance:
            return i, None
        elif i < len(instances) - 1 and n > instance and n < instances[i + 1]:
            return i, i
    return i, None


def inter_cal(a, b, c, d, e):
    return (e - a) * (c - d) / (a - b) + c

def reg_format(a):
    reg = '%.2f' % round(a, 2)
    if reg[-1] == 0:
        return reg[:-1]
    else:
        return reg


def interpolate(n, instances, price):
    # Write your code here
    pop_laps(instances, price)
    double_sort(instances, price)

    l = len(instances)
    if l == 1:
        return reg_format(price[0])

    i, j = judge_index(instances, n)
    if j is None:
        if i < len(instances) - 1:
            return reg_format(price[i])

        else:
            return reg_format(inter_cal(instances[l - 2], instances[l - 1], price[l - 2], price[l - 1], n) + 10e-5)

    elif j == -1:
        return reg_format(inter_cal(instances[0], instances[1], price[0], price[1], n) + 10e-5)

    else:
        return reg_format(inter_cal(instances[j], instances[j + 1], price[j], price[j + 1], n) + 10e-5)





n = 2
ins = [10, 25, 50, 100, 500]
pr = [0.0, 0.0, 0.0, 0.0, 54.25]

interpolate(n, ins, pr)



#%%
import copy
def build_matrix(n, n_1, matrix, max_sum):
    a = [0] * n
    b = list()
    c = list()
    for i in range(n):
        b.append(a)
    for i in range(n):
        c.append(b)
    c[0] = matrix

    for item in matrix:
        for it in item:
            if it > max_sum:
                return 0

    for k in range(1,n):
        tmp = copy.deepcopy(b)
        for i in range(k, n):
            tmp_1 = copy.deepcopy(a)
            for j in range(k, n):
                w = c[k-1][i-1][j-1] + sum(c[0][i][j-k:j+1]) + sum([item[j] for item in c[0][i-k:i+1]]) - c[0][i][j]
                tmp_1[j] = w
            tmp[i] = tmp_1
        mark = 1
        for item in tmp:
            for ite in item:
                if ite > max_sum:
                    mark = 0
        if mark == 0:
            return k
        c[k] = tmp
    return n

if __name__ == '__main__':
    matrix = [[1,2,3,4,1],
              [5,6,7,8,1],
              [9,10,11,12,1],
              [13,14,15,16,1],
              [13,14,15,16,1]]
    n = 4
    print(build_matrix(n,n,matrix,150))






#%%

import copy
def build_matrix(n, n_1, matrix, max_sum):
    a = [0] * n
    b = list()
    c = list()
    for i in range(n):
        b.append(a)
    for i in range(n):
        c.append(b)
    c[0] = matrix

    for item in matrix:
        for it in item:
            if it > max_sum:
                return 0
    for k in range(1,n):
        tmp = copy.deepcopy(b)
        for i in range(k, n):
            tmp_1 = [0] * n
            for j in range(k, n):
                w = c[k-1][i-1][j-1] + sum(c[0][i][j-k:j+1]) + sum([item[j] for item in c[0][i-k:i+1]]) - c[0][i][j]
                if w > max_sum:
                    return k
                tmp_1[j] = w
            tmp[i] = tmp_1
        c[k] = tmp
    return n

def largestSubgrid(grid, maxSum):
    # Write your code here
    n = len(grid)
    m = build_matrix(n, n, grid, maxSum)
    return m

'''

























