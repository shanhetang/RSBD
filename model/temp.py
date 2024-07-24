import numpy as np
import pandas as pd

# 示例数据
user_item_matrix = pd.DataFrame({
    'item1': [5, 0, 3, 0],
    'item2': [4, 0, 0, 2],
    'item3': [0, 1, 2, 0],
    'item4': [0, 0, 4, 5]
})
user_similarity = np.array([
    [1.0, 0.3, 0.4, 0.2],
    [0.3, 1.0, 0.1, 0.4],
    [0.4, 0.1, 1.0, 0.5],
    [0.2, 0.4, 0.5, 1.0]
])

user_ids = user_item_matrix.index
print("User IDs:", user_ids)

# 获取前n个相似度最高的用户
n = 2 # 取前10个相似用户
for i, user_id in enumerate(user_ids):
    user_vector = user_item_matrix.iloc[i]
    similar_users = user_similarity[i].argsort()[::-1][1:n + 1]  # 取前n个最相似的用户（排除自己）

    # 遍历item，获取top-k最高得分的item
    for item_id in user_item_matrix.columns:
        if user_vector[item_id] == 0:  # 用户未交互过的物品
            similar_users_ratings = user_item_matrix.iloc[similar_users][item_id]
            similar_users_similarities = user_similarity[i, similar_users]
            print(similar_users)
            # 过滤掉相似用户中未评分的情况
            mask = similar_users_ratings > 0
            if mask.sum() == 0:
                continue

            filtered_similar_users_ratings = similar_users_ratings[mask]
            filtered_similar_users_similarities = similar_users_similarities[mask]

            print("Similar users' ratings for item", item_id, ":", filtered_similar_users_ratings)
            print("Similar users' similarities for item", item_id, ":", filtered_similar_users_similarities)
            print("Sum of similar users' ratings for item", item_id, ":", np.sum(filtered_similar_users_ratings))

            # 加权平均得分
            score = np.sum(filtered_similar_users_ratings * filtered_similar_users_similarities) / np.sum(filtered_similar_users_similarities)
            print("Predicted score for item", item_id, ":", score)
            break
    break

