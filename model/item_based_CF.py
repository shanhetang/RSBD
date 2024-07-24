"""
基于用户/物品的CF
    1、拆分数据集
        1.1 将id映射到指定范围
        1.2 划分训练集和测试集
    2、根据训练集计算相似度矩阵
        2.1 生成user-item矩阵
        2.2 计算用户相似度
        2.3 计算物品相似度
    3、推荐
        3.1 基于用户的top-k推荐
        3.2 基于物品的top-k推荐
    4、进行评估
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from collections import defaultdict

from tqdm import tqdm


class CFRecommendationSystem:
    def __init__(self, data):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.u_recommendations = defaultdict(list)
        self.i_recommendations = defaultdict(list)

    # 2、根据训练集计算相似度矩阵
    # 2.1 生成user-item矩阵
    def build_user_item_matrix(self, data, user_id_map, item_id_map):
        self.user_item_matrix = np.zeros((len(user_id_map), len(item_id_map)))
        for line in data.itertuples():
            self.user_item_matrix[line[1] - 1, line[2] - 1] = line[3]
        self.user_item_matrix = pd.DataFrame(self.user_item_matrix, index=user_id_map.values(),
                                             columns=item_id_map.values())

    # 2.2 计算用户相似度
    def compute_user_similarity(self, user_item_matrix, method="cos"):
        if method == 'cos':
            similarity = cosine_similarity(user_item_matrix)
        else:  # method == 'pearson':
            similarity = 1 - pairwise_distances(user_item_matrix, metric='correlation')
        self.user_similarity = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # 2.3 计算物品相似度
    def compute_item_similarity(self, user_item_matrix, method="cos"):
        if method == 'cos':
            similarity = cosine_similarity(user_item_matrix.T)
        else:  # method == 'pearson':
            similarity = 1 - pairwise_distances(user_item_matrix.T, metric='correlation')
        self.item_similarity = pd.DataFrame(similarity, index=user_item_matrix.T.index,
                                            columns=user_item_matrix.T.index)

    # 3、推荐
    def get_top_n_recommendations(self, top_k=10, method="user"):
        # 3.1 基于用户的top-k推荐
        if method == "user":
            user_ids = self.user_item_matrix.index
            for user_id in tqdm(user_ids, desc="Processing users"):
                user_vector = self.user_item_matrix.iloc[user_id]
                similar_users = self.user_similarity[user_id].argsort()[::-1][1:top_k + 1]  # 取前K个最相似的用户（除了自己）
                # 遍历item，获取top-k最高得分的item
                for item_id in self.user_item_matrix.columns:
                    if user_vector[item_id] == 0:  # 用户未交互过的物品
                        similar_users_ratings = self.user_item_matrix.iloc[similar_users, item_id]
                        similar_users_similarities = self.user_similarity.iloc[user_id, similar_users]
                        # 加权平均得分
                        score = np.sum(similar_users_ratings * similar_users_similarities) / np.sum(
                            similar_users_similarities)
                        if score > 0:
                            self.u_recommendations[user_id].append((item_id, score))
                self.u_recommendations[user_id] = sorted(self.u_recommendations[user_id], key=lambda x: x[1],
                                                         reverse=True)[:top_k]
        # 3.2 基于物品的top-k推荐
        elif method == "item":
            user_ids = self.user_item_matrix.index
            for user_id in tqdm(user_ids, desc="Processing users"):
                # 获取交互过的物品和候选物品
                user_vector = self.user_item_matrix.iloc[user_id]
                interacted_items = user_vector[user_vector > 0].index
                candidate_items = self.user_item_matrix.columns[~self.user_item_matrix.columns.isin(interacted_items)]

                # 计算得分
                item_scores = defaultdict(float)
                for item in candidate_items:
                    for interacted_item in interacted_items:
                        item_scores[item] += self.item_similarity.iloc[item, interacted_item] * user_vector[interacted_item]

                top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                self.i_recommendations[user_id] = top_items

    # 4、评估
    def evaluate(self, test_data, top_k=10, method="user"):
        precision = defaultdict(float)
        recall = defaultdict(float)
        if method == "user":
            recommendations = self.u_recommendations
        else:  # if method == "item":
            recommendations = self.i_recommendations

        for user, user_recs in recommendations.items():
            user_test = set(test_data[test_data['user_id'] == user]['item_id'])
            recs = set([item for item, _ in user_recs[:top_k]])

            if len(recs) > 0:
                precision[user] = len(recs & user_test) / len(recs)
            if len(user_test) > 0:
                recall[user] = len(recs & user_test) / len(user_test)

        avg_precision = np.mean(list(precision.values()))
        avg_recall = np.mean(list(recall.values()))

        return avg_precision, avg_recall


# 1、拆分数据集
def load_data_and_map_ids(file_path, test_size=0.3):
    # 读取数据
    data = pd.read_csv(file_path, sep=' ', header=None, names=['user_id', 'item_id', 'click'])

    # 1.1 将id映射到指定范围
    user_id_map = {id: i for i, id in enumerate(data['user_id'].unique())}
    item_id_map = {id: i for i, id in enumerate(data['item_id'].unique())}

    data['user_id'] = data['user_id'].map(user_id_map)
    data['item_id'] = data['item_id'].map(item_id_map)

    # 1.2 划分训练集和测试集
    train_data = []
    test_data = []
    for user, group in data.groupby('user_id'):
        n_test_items = max(1, int(len(group) * test_size))
        test_items = group.sample(n=n_test_items)
        train_items = group.drop(test_items.index)
        train_data.append(train_items)
        test_data.append(test_items)

    return data, user_id_map, item_id_map, pd.concat(train_data), pd.concat(test_data)


# 将映射后的ID转换回原始ID
def convert_recommendations_to_original_ids(recommendations, user_id_map, item_id_map):
    original_recommendations = {}
    reverse_user_id_map = {v: k for k, v in user_id_map.items()}
    reverse_item_id_map = {v: k for k, v in item_id_map.items()}

    for user, recs in recommendations.items():
        original_user_id = reverse_user_id_map[user]
        original_recs = [(reverse_item_id_map[item], score) for item, score in recs]
        original_recommendations[original_user_id] = original_recs

    return original_recommendations


if __name__ == "__main__":
    # 加载数据并拆分训练集和测试集
    data, user_id_map, item_id_map, train_data, test_data = load_data_and_map_ids('../data/training.txt')

    # 构建user-item交互矩阵

