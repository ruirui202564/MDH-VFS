import numpy as np
import pandas as pd
from scipy.io import loadmat
import random


def load_mat_data(file_path):
    """加载.mat格式的数据集"""
    mat_data = loadmat(file_path)
    # 假设数据存储在'data'和'label'键中，根据实际情况调整
    # 常见的键名有: 'X', 'y', 'data', 'label', 'features', 'targets'

    data = mat_data['data']
    X = data[:, 1:data.shape[1]]
    X = X.astype(float)

    Y = data[:, 0]
    Y = Y.reshape(-1, 1)
    Y = Y.astype(int)

    return X, Y


def generate_batch1(X, y, num_samples=2000, feature_ratio=0.7, start_idx=0):
    """
    生成batch1：使用原始数据的前2000个样本，只使用70%的特征

    参数:
    - X: 特征矩阵
    - y: 标签向量
    - num_samples: 样本数量
    - feature_ratio: 使用的特征比例
    - start_idx: 起始样本索引
    """
    total_features = X.shape[1]
    num_used_features = int(total_features * feature_ratio)

    # 随机选择70%的特征
    all_features = list(range(total_features))
    used_features = sorted(random.sample(all_features, num_used_features))
    remaining_features = sorted(list(set(all_features) - set(used_features)))

    # 提取数据
    end_idx = start_idx + num_samples
    X_batch = X[start_idx:end_idx, :][:, used_features]
    y_batch = y[start_idx:end_idx]

    # 计算每个特征的标准差（用于后续batch）
    feature_std = {}
    for i, feat_idx in enumerate(used_features):
        feature_std[feat_idx] = np.std(X_batch[:, i])

    # 合并数据
    data_batch = np.hstack((X_batch, y_batch.reshape(-1, 1)))

    return data_batch, used_features, remaining_features, feature_std


def change_distribution(X, y, used_features, feature_std, ratio_change=0.2, k=1.0,
                        start_idx=2000, num_samples=2000, changed_features_prev=None):
    """
    batch2: 只改变分布

    参数:
    - X: 原始特征矩阵
    - y: 标签向量
    - used_features: 当前使用的特征列表
    - feature_std: 上一个batch中特征的标准差字典
    - ratio_change: 改变分布的特征比例
    - k: 分布变化系数
    - start_idx: 起始样本索引
    - num_samples: 样本数量
    - changed_features_prev: 之前batch中已改变分布的特征及其k值
    """
    end_idx = start_idx + num_samples
    # 确保至少改变1个特征
    num_change = max(1, int(np.round(len(used_features) * ratio_change)))
    num_change = min(num_change, len(used_features))

    # 如果之前没有改变过特征，则随机选择
    if changed_features_prev is None:
        changed_features_prev = {}

    # 随机选择需要改变分布的特征
    changed_features = random.sample(used_features, num_change)

    # 提取原始数据
    X_batch = X[start_idx:end_idx, :][:, used_features].copy()
    y_batch = y[start_idx:end_idx]

    # 更新特征标准差
    new_feature_std = {}
    for i, feat_idx in enumerate(used_features):
        new_feature_std[feat_idx] = np.std(X_batch[:, i])

    # 改变分布
    distribution_changes = {}
    for feat_idx in used_features:
        feat_pos = used_features.index(feat_idx)

        if feat_idx in changed_features:
            # 如果该特征之前已经改变过，使用相同的k值
            if feat_idx in changed_features_prev:
                k_use = changed_features_prev[feat_idx]
            else:
                k_use = k

            # 应用分布变化: x_new = x + k * σ
            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma

            distribution_changes[feat_idx] = k_use
        elif feat_idx in changed_features_prev:
            # 如果该特征之前改变过但这次没被选中，仍然应用之前的变化
            k_use = changed_features_prev[feat_idx]
            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma

    # 合并所有改变记录
    all_changed = changed_features_prev.copy()
    all_changed.update(distribution_changes)

    data_batch = np.hstack((X_batch, y_batch.reshape(-1, 1)))

    change_info = {
        'type': 'distribution_change',
        'changed_features': list(distribution_changes.keys())
    }

    return data_batch, used_features, new_feature_std, change_info, all_changed


def change_space(X, y, used_features, remaining_features, feature_std,
                 ratio_silent=0.1, ratio_new=0.2, start_idx=4000, num_samples=2000):
    """
    batch3: 只改变特征空间

    参数:
    - ratio_silent: 移除特征的比例
    - ratio_new: 添加新特征的比例（从剩余30%中选择）
    """
    end_idx = start_idx + num_samples

    # 确定移除和添加的特征，确保数量不为0
    num_silent = max(1, int(np.round(len(used_features) * ratio_silent)))
    num_silent = min(num_silent, len(used_features) - 1)  # 至少保留1个特征
    silent_features = random.sample(used_features, num_silent)

    # 计算可以添加的新特征数量，确保不为0
    num_new = max(1, int(np.round(len(used_features) * ratio_new)))
    num_new = min(num_new, len(remaining_features))
    new_features = random.sample(remaining_features, num_new) if num_new > 0 else []

    # 更新特征列表
    active_features = [f for f in used_features if f not in silent_features]
    updated_used_features = sorted(active_features + new_features)
    updated_remaining = [f for f in remaining_features if f not in new_features]

    # 提取数据
    X_batch = X[start_idx:end_idx, :][:, updated_used_features]
    y_batch = y[start_idx:end_idx]

    # 更新特征标准差
    new_feature_std = {}
    for i, feat_idx in enumerate(updated_used_features):
        new_feature_std[feat_idx] = np.std(X_batch[:, i])

    data_batch = np.hstack((X_batch, y_batch.reshape(-1, 1)))

    change_info = {
        'type': 'space_change',
        'silent_features': silent_features,
        'new_features': new_features
    }

    return data_batch, updated_used_features, updated_remaining, new_feature_std, change_info


def change_space_distribution(X, y, used_features, remaining_features, feature_std,
                              ratio_silent=0.1, ratio_new=0.2, ratio_change=0.2,
                              k=1.0, start_idx=6000, num_samples=2000,
                              changed_features_prev=None):
    """
    batch4: 特征空间和分布都变化
    """
    end_idx = start_idx + num_samples

    if changed_features_prev is None:
        changed_features_prev = {}

    # 先进行特征空间变化，确保数量不为0
    num_silent = max(1, int(np.round(len(used_features) * ratio_silent)))
    num_silent = min(num_silent, len(used_features) - 1)  # 至少保留1个特征
    silent_features = random.sample(used_features, num_silent)

    # 使用所有剩余特征作为新特征
    num_new = len(remaining_features)
    new_features = remaining_features.copy()

    active_features = [f for f in used_features if f not in silent_features]
    updated_used_features = sorted(active_features + new_features)

    # 从active特征中选择需要改变分布的特征，确保数量不为0
    num_change = max(1, int(np.round(len(active_features) * ratio_change)))
    num_change = min(num_change, len(active_features))
    changed_features = random.sample(active_features, num_change)

    # 提取数据
    X_batch = X[start_idx:end_idx, :][:, updated_used_features].copy()
    y_batch = y[start_idx:end_idx]

    # 更新特征标准差
    new_feature_std = {}
    for i, feat_idx in enumerate(updated_used_features):
        new_feature_std[feat_idx] = np.std(X_batch[:, i])

    # 改变分布
    distribution_changes = {}
    for feat_idx in updated_used_features:
        if feat_idx not in active_features:
            continue

        feat_pos = updated_used_features.index(feat_idx)

        if feat_idx in changed_features:
            # 如果之前改变过，使用相同的k值
            if feat_idx in changed_features_prev:
                k_use = changed_features_prev[feat_idx]
            else:
                k_use = k

            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma
            distribution_changes[feat_idx] = k_use
        elif feat_idx in changed_features_prev:
            # 之前改变过但这次没被选中，仍然应用之前的变化
            k_use = changed_features_prev[feat_idx]
            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma

    all_changed = changed_features_prev.copy()
    all_changed.update(distribution_changes)

    data_batch = np.hstack((X_batch, y_batch.reshape(-1, 1)))

    change_info = {
        'type': 'space_distribution_change',
        'silent_features': silent_features,
        'new_features': new_features,
        'changed_features': list(distribution_changes.keys())
    }

    return data_batch, updated_used_features, [], new_feature_std, change_info, all_changed


def align_feature_spaces(batches, feature_lists, total_features):
    """对齐所有batch的特征空间"""
    aligned_batches = []

    for batch, features in zip(batches, feature_lists):
        labels = batch[:, -1:]
        data = batch[:, :-1]

        # 创建对齐后的特征矩阵
        num_samples = batch.shape[0]
        aligned_data = np.full((num_samples, total_features), np.nan)

        # 填充数据
        for i, feat_idx in enumerate(features):
            aligned_data[:, feat_idx] = data[:, i]

        aligned_batch = np.hstack((aligned_data, labels))
        aligned_batches.append(aligned_batch)

    return aligned_batches


def export_to_excel(aligned_batches, total_features, change_info_list, output_file):
    """导出到Excel，包含All_Data、Change_Info和Distribution_Change_Details三个表"""
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 创建列名
        feature_columns = [f'feature_{i}' for i in range(total_features)]
        columns = feature_columns + ['label']

        # 合并所有batch到All_Data表
        all_data = []
        for batch_idx, batch in enumerate(aligned_batches):
            batch_df = pd.DataFrame(batch, columns=columns)
            all_data.append(batch_df)

        result_df = pd.concat(all_data, ignore_index=True)
        result_df.to_excel(writer, sheet_name='All_Data', index=False)

        # 创建Change_Info表
        change_data = []
        for batch_idx, info in enumerate(change_info_list):
            batch_num = batch_idx + 2  # batch2开始才有变化
            change_dict = {
                'batch_id': batch_num,
                'change_type': info['type']
            }

            # 根据变化类型添加不同的信息
            if 'silent_features' in info:
                change_dict['silent_features'] = str(info['silent_features'])
            else:
                change_dict['silent_features'] = ''

            if 'new_features' in info:
                change_dict['new_features'] = str(info['new_features'])
            else:
                change_dict['new_features'] = ''

            if 'changed_features' in info:
                change_dict['changed_distributions'] = str(info['changed_features'])
            else:
                change_dict['changed_distributions'] = ''

            change_data.append(change_dict)

        if change_data:
            change_df = pd.DataFrame(change_data)
            change_df.to_excel(writer, sheet_name='Change_Info', index=False)

        # 创建Distribution_Change_Details表
        distribution_change_details = []
        for batch_idx, info in enumerate(change_info_list):
            batch_num = batch_idx + 2  # batch2开始才有变化
            if 'changed_features' in info and info['changed_features']:
                for feat_idx in info['changed_features']:
                    distribution_change_details.append({
                        'batch_id': batch_num,
                        'feature_id': feat_idx
                    })

        if distribution_change_details:
            dist_change_df = pd.DataFrame(distribution_change_details)
            dist_change_df.to_excel(writer, sheet_name='Distribution_Change_Details', index=False)


if __name__ == "__main__":
    # 参数设置
    # 'E:/论文/7-特征增量/8-INS三审意见修改/OLSF/data/spambase.mat'
    # 'E:/论文/10-概念漂移+梯形数据流/6-drift完善后/0-数据及代码/Dataset_UCI_drift/Trap_Cap/magic04.mat'
    # 'E:/论文/10-概念漂移+梯形数据流/6-drift完善后/4-IPM一审意见修改（12.7-1.6）/Dataset_add/Trap_Cap/electricity.mat'
    mat_file_path = 'E:/论文/10-概念漂移+梯形数据流/6-drift完善后/4-IPM一审意见修改（12.7-1.6）/Dataset_add/Trap_Cap/electricity.mat'  # 替换为实际的.mat文件路径
    num_samples = 2000
    k = 5.0  # 分布变化系数

    # 加载数据
    print("正在加载数据...")
    X, y = load_mat_data(mat_file_path)
    total_samples, total_features = X.shape
    print(f"数据集形状: {X.shape}, 标签形状: {y.shape}")

    # 检查样本数量是否足够
    required_samples =  num_samples * 4
    if total_samples < required_samples:
        print(f"警告: 数据集只有{total_samples}个样本，少于所需的{required_samples}个")

    # 生成batch1
    print("\n生成Batch 1...")
    batch1, used_features1, remaining_features1, feature_std1 = generate_batch1(
        X, y, num_samples=num_samples, feature_ratio=0.7, start_idx=0
    )
    print(f"Batch 1: 使用{len(used_features1)}个特征, 剩余{len(remaining_features1)}个特征")

    batches = [batch1]
    feature_lists = [used_features1]
    change_info_list = []

    # 生成batch2
    print("\n生成Batch 2...")
    batch2, used_features2, feature_std2, change_info2, changed_dist2 = change_distribution(
        X, y, used_features1, feature_std1,
        ratio_change=0.2, k=k, start_idx=2000, num_samples=num_samples
    )
    batches.append(batch2)
    feature_lists.append(used_features2)
    change_info_list.append(change_info2)
    print(f"Batch 2: {len(change_info2['changed_features'])}个特征分布发生变化")

    # 生成batch3
    print("\n生成Batch 3...")
    batch3, used_features3, remaining_features3, feature_std3, change_info3 = change_space(
        X, y, used_features2, remaining_features1, feature_std2,
        ratio_silent=0.1, ratio_new=0.2, start_idx=4000, num_samples=num_samples
    )
    batches.append(batch3)
    feature_lists.append(used_features3)
    change_info_list.append(change_info3)
    print(f"Batch 3: 移除{len(change_info3['silent_features'])}个特征, 添加{len(change_info3['new_features'])}个特征")

    # 生成batch4
    print("\n生成Batch 4...")
    batch4, used_features4, remaining_features4, feature_std4, change_info4, changed_dist4 = change_space_distribution(
        X, y, used_features3, remaining_features3, feature_std3,
        ratio_silent=0.1, ratio_new=1.0, ratio_change=0.2,
        k=k, start_idx=6000, num_samples=num_samples,
        changed_features_prev=changed_dist2
    )
    batches.append(batch4)
    feature_lists.append(used_features4)
    change_info_list.append(change_info4)
    print(
        f"Batch 4: 移除{len(change_info4['silent_features'])}个特征, 添加{len(change_info4['new_features'])}个特征, {len(change_info4['changed_features'])}个特征分布发生变化")

    # 对齐特征空间
    print("\n对齐特征空间...")
    aligned_batches = align_feature_spaces(batches, feature_lists, total_features)

    # 导出Excel
    output_file = f'realData_new/electricity_{int(k)}_.xlsx'
    print(f"\n导出数据到 {output_file}...")
    export_to_excel(aligned_batches, total_features, change_info_list, output_file)

    print("\n完成!")
    print(f"总特征数: {total_features}")
    for i, features in enumerate(feature_lists):
        print(f"Batch {i + 1} 使用特征数: {len(features)}")