import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

# 日本語フォントを読み込む
mpl.rcParams['font.family'] = 'Hiragino Sans'

# CSVファイルからデータを読み込む
data = pd.read_csv("gure_new.csv", encoding="utf-8")

# 走行距離と価格の平均値を計算
# 価格の平均値を計算
data["price_avg"] = data["価格(万円)"].str.replace("~", "～").str.split("～").apply(lambda x: (int(x[0]) + int(x[1])) / 2)

# 散布図を作成
plt.figure(figsize=(10, 6))
plt.scatter(data["走行"], data["price_avg"])
plt.xlabel("走行距離 (万km)")
plt.ylabel("価格 (万円)")
plt.title("ホンダ S2000の走行距離と価格の散布図")
plt.grid()
plt.show()

# 相関係数を計算
correlation = data["走行"].corr(data["price_avg"])

# 相関図を作成
plt.figure(figsize=(10, 6))
plt.scatter(data["走行"], data["price_avg"])
plt.xlabel("走行距離 (万km)")
plt.ylabel("価格 (万円)")
plt.title(f"ホンダ S2000の走行距離と価格の相関図\n相関係数: {correlation:.2f}")
plt.grid()

# 回帰直線を追加
m, b = np.polyfit(data["走行"], data["price_avg"], 1)
plt.plot(data["走行"], m * data["走行"] + b, color="red", label="回帰直線")
plt.legend()
plt.show()