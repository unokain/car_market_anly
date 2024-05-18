import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 日本語フォントを読み込む
mpl.rcParams['font.family'] = 'Hiragino Sans'

# CSVファイルからデータを読み込む
data = pd.read_csv("blue_s2000.csv", encoding="utf-8")

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

# 多項式回帰のための特徴量変換
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(data[["走行"]])

# 線形回帰モデルの適用
lin_reg = LinearRegression()
lin_reg.fit(X_poly, data["price_avg"])

# 回帰曲線の描画
X_new = np.linspace(data["走行"].min(), data["走行"].max(), 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X_new, y_new, color="red", label="多項式回帰曲線")

plt.legend()
plt.show()