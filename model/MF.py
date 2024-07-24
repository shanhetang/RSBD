import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


# 加载数据
df = pd.read_csv('../data/training.txt', sep=' ', header=None, names=['user_id', 'item_id', 'click'])


def mf_top_n(df, user_id, n=5, k=20):
    # 创建用户-物品交互矩阵
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='click').fillna(0)

    # 执行SVD
    U, sigma, Vt = svds(user_item_matrix.values, k=k)

    # 重建矩阵
    sigma_diag = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt)

    # 将预测结果转换为DataFrame
    preds_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

    # 获取用户未交互的物品
    user_items = user_item_matrix.loc[user_id]
    uninteracted_items = user_items[user_items == 0].index

    # 对未交互的物品进行排序
    recommendations = preds_df.loc[user_id, uninteracted_items].sort_values(ascending=False)

    # 返回top-N推荐
    return recommendations.head(n).index.tolist()


# 示例：为用户ID为1的用户推荐5个物品
print("矩阵分解推荐结果:", mf_top_n(df, 1, 5))