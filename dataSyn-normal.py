import numpy as np
import random
import pandas as pd

def generate_batch1(num_features, num_samples, distribution='normal_5'):
    data = np.zeros((num_samples, num_features))
    fea_distribution = {}
    if distribution == 'normal_5':
        data[:, :] = np.random.normal(0, 1, (num_samples, num_features))
        for i in range(num_features):
            fea_distribution[i] = ('normal_5', 0, 1)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, num_features))
    extra_data[:, :] = np.random.normal(0, 1, (extra_samples, num_features))

    means_list = [dist_info[1] for dist_info in fea_distribution.values()]
    threshold = np.sum(means_list)

    f = np.sum(extra_data[:, :], axis=1)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return generate_batch1(num_features, num_samples, distribution)

    selected_idx_0 = np.random.choice(idx_label_0, num_samples // 2, replace=False)
    selected_idx_1 = np.random.choice(idx_label_1, num_samples // 2, replace=False)

    selected_idx = np.concatenate([selected_idx_0, selected_idx_1])
    np.random.shuffle(selected_idx)

    balanced_data = extra_data[selected_idx]
    balanced_labels = extra_labels[selected_idx]

    final_data = np.hstack((balanced_data, balanced_labels.reshape(-1, 1)))

    uni_fea = list(range(num_features))
    return final_data, fea_distribution, uni_fea

def generate_batch(change, num_samples, uni_fea_old, fea_distribution, ratio_change, ratio_silent, ratio_new):
    if change == 1: 
        data, fea_distribution_new, change_info = change_distribution(num_samples, uni_fea_old, fea_distribution,
                                                                      ratio_change)
        return data, uni_fea_old, fea_distribution_new, change_info
    elif change == 2:
        data, uni_fea, fea_distribution_new, change_info = change_space(num_samples, uni_fea_old, ratio_silent,
                                                                        ratio_new, fea_distribution)
    elif change == 3: 
        data, uni_fea, fea_distribution_new, change_info = change_space_distribution(num_samples, uni_fea_old,
                                                                                     ratio_silent, ratio_new,
                                                                                     ratio_change, fea_distribution)
    elif change == 0:
        data, change_info = no_change(num_samples, uni_fea_old, fea_distribution)
        return data, uni_fea_old, fea_distribution, change_info
    return data, uni_fea, fea_distribution_new, change_info

def no_change(num_samples, uni_fea_old, fea_distribution):
    num_fea_old = len(uni_fea_old)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, num_fea_old))

    for i in range(num_fea_old):
        distribution_type, mean, variance = fea_distribution[uni_fea_old[i]]
        extra_data[:, i] = np.random.normal(mean, variance, extra_samples)

    f = np.sum(extra_data, axis=1)
    means_list = [fea_distribution[idx][1] for idx in uni_fea_old]
    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_distribution(num_samples, uni_fea_old, fea_distribution, ratio_change)

    selected_idx_0 = np.random.choice(idx_label_0, num_samples // 2, replace=False)
    selected_idx_1 = np.random.choice(idx_label_1, num_samples // 2, replace=False)

    selected_idx = np.concatenate([selected_idx_0, selected_idx_1])
    np.random.shuffle(selected_idx)

    balanced_data = extra_data[selected_idx]
    balanced_labels = extra_labels[selected_idx]

    final_data = np.hstack((balanced_data, balanced_labels.reshape(-1, 1)))

    change_info = {
        'type': 'no_change'
    }

    return final_data, change_info

def change_distribution(num_samples, uni_fea_old, fea_distribution, ratio_change):
    num_fea_old = len(uni_fea_old)
    uni_fea_old_list = uni_fea_old.tolist() if isinstance(uni_fea_old, np.ndarray) else list(uni_fea_old)
    num_change = int(np.round(num_fea_old * ratio_change))
    change = random.sample(uni_fea_old_list, num_change)

    distribution_changes = {}

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, num_fea_old))

    for i in range(num_fea_old):
        distribution_type, mean, variance = fea_distribution[uni_fea_old[i]]
        if uni_fea_old[i] in change:
            if distribution_type == 'normal_5':
                mean_new = random.randint(1, 5)
                variance_new = random.randint(1, 5)
                distribution_changes[uni_fea_old[i]] = {
                    'old': (distribution_type, mean, variance),
                    'new': (distribution_type, mean + mean_new, variance + variance_new)
                }
                fea_distribution[uni_fea_old[i]] = ('normal_5', mean + mean_new, variance + variance_new)
                extra_data[:, i] = np.random.normal(mean + mean_new, variance + variance_new, extra_samples)
        else:
            extra_data[:, i] = np.random.normal(mean, variance, extra_samples)

    f = np.sum(extra_data, axis=1)
    means_list = [fea_distribution[idx][1] for idx in uni_fea_old]
    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_distribution(num_samples, uni_fea_old, fea_distribution, ratio_change)

    selected_idx_0 = np.random.choice(idx_label_0, num_samples // 2, replace=False)
    selected_idx_1 = np.random.choice(idx_label_1, num_samples // 2, replace=False)

    selected_idx = np.concatenate([selected_idx_0, selected_idx_1])
    np.random.shuffle(selected_idx)

    balanced_data = extra_data[selected_idx]
    balanced_labels = extra_labels[selected_idx]

    final_data = np.hstack((balanced_data, balanced_labels.reshape(-1, 1)))

    change_info = {
        'type': 'distribution_change',
        'changed_features': change,
        'distribution_changes': distribution_changes
    }

    return final_data, fea_distribution, change_info

def change_space(num_samples, uni_fea_old, ratio_silent, ratio_new, fea_distribution, new_distribution='normal_5'):
    num_fea_old = len(uni_fea_old)
    uni_fea_old_list = uni_fea_old.tolist() if isinstance(uni_fea_old, np.ndarray) else list(uni_fea_old)
    num_silent = int(np.round(num_fea_old * ratio_silent))
    silent = random.sample(uni_fea_old_list, num_silent)
    num_new = int(np.round(num_fea_old * ratio_new))
    last_fea = uni_fea_old[-1]
    new = [last_fea + i for i in range(1, num_new + 1)]
    new = np.array(new)
    active = np.setdiff1d(uni_fea_old, silent)
    uni_fea = np.union1d(uni_fea_old, new)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, len(uni_fea)))
    extra_data[:, silent] = np.nan

    for i in range(len(active)):
        distribution_type, mean, variance = fea_distribution[active[i]]
        if distribution_type == 'normal_5':
            extra_data[:, active[i]] = np.random.normal(mean, variance, extra_samples)

    for i in range(len(new)):
        if new_distribution == 'normal_5':
            extra_data[:, new[i]] = np.random.normal(0, 1, extra_samples)
            fea_distribution[new[i]] = ('normal_5', 0, 1)

    current = np.concatenate((active, new))
    f = np.nansum(extra_data[:, current], axis=1)
    means_list = [fea_distribution[idx][1] for idx in current]
    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_space(num_samples, uni_fea_old, ratio_silent, ratio_new, fea_distribution, new_distribution)

    selected_idx_0 = np.random.choice(idx_label_0, num_samples // 2, replace=False)
    selected_idx_1 = np.random.choice(idx_label_1, num_samples // 2, replace=False)

    selected_idx = np.concatenate([selected_idx_0, selected_idx_1])
    np.random.shuffle(selected_idx)

    balanced_data = extra_data[selected_idx]
    balanced_labels = extra_labels[selected_idx]

    final_data = np.hstack((balanced_data, balanced_labels.reshape(-1, 1)))

    change_info = {
        'type': 'space_change',
        'silent_features': silent,
        'new_features': new.tolist()
    }

    return final_data, uni_fea, fea_distribution, change_info

def change_space_distribution(num_samples, uni_fea_old, ratio_silent, ratio_new, ratio_change, fea_distribution,
                              new_distribution='normal_5'):
    num_fea_old = len(uni_fea_old)
    uni_fea_old_list = uni_fea_old.tolist() if isinstance(uni_fea_old, np.ndarray) else list(uni_fea_old)
    num_silent = int(np.round(num_fea_old * ratio_silent))
    silent = random.sample(uni_fea_old_list, num_silent)
    num_new = int(np.round(num_fea_old * ratio_new))
    last_fea = uni_fea_old[-1]
    new = [last_fea + i for i in range(1, num_new + 1)]
    new = np.array(new)
    active = np.setdiff1d(uni_fea_old, silent)
    uni_fea = np.union1d(uni_fea_old, new)

    active_list = active.tolist() if isinstance(active, np.ndarray) else list(active)
    num_change = int(np.round(len(active) * ratio_change))
    change = random.sample(active_list, num_change)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, len(uni_fea)))
    extra_data[:, silent] = np.nan

    distribution_changes = {}

    for i in range(len(uni_fea)):
        if uni_fea[i] in new:
            if new_distribution == 'normal_5':
                extra_data[:, uni_fea[i]] = np.random.normal(0, 1, extra_samples)
                fea_distribution[uni_fea[i]] = ('normal_5', 0, 1)
        elif uni_fea[i] in active:
            distribution_type, mean, variance = fea_distribution[uni_fea[i]]
            if uni_fea[i] in change:
                if distribution_type == 'normal_5':
                    mean_new = random.randint(1, 5)
                    variance_new = random.randint(1, 5)
                    distribution_changes[uni_fea[i]] = {
                        'old': (distribution_type, mean, variance),
                        'new': (distribution_type, mean + mean_new, variance + variance_new)
                    }
                    fea_distribution[uni_fea[i]] = ('normal_5', mean + mean_new, variance + variance_new)
                    extra_data[:, uni_fea[i]] = np.random.normal(mean + mean_new, variance + variance_new,
                                                                 extra_samples)
            else:
                if distribution_type == 'normal_5':
                    extra_data[:, uni_fea[i]] = np.random.normal(mean, variance, extra_samples)

    current = np.concatenate((active, new))
    f = np.nansum(extra_data[:, current], axis=1)
    means_list = [fea_distribution[idx][1] for idx in current]
    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_space_distribution(num_samples, uni_fea_old, ratio_silent, ratio_new, ratio_change,
                                         fea_distribution, new_distribution)

    selected_idx_0 = np.random.choice(idx_label_0, num_samples // 2, replace=False)
    selected_idx_1 = np.random.choice(idx_label_1, num_samples // 2, replace=False)

    selected_idx = np.concatenate([selected_idx_0, selected_idx_1])
    np.random.shuffle(selected_idx)

    balanced_data = extra_data[selected_idx]
    balanced_labels = extra_labels[selected_idx]

    final_data = np.hstack((balanced_data, balanced_labels.reshape(-1, 1)))

    change_info = {
        'type': 'space_distribution_change',
        'silent_features': silent,
        'new_features': new.tolist(),
        'changed_distributions': change,
        'distribution_changes': distribution_changes
    }

    return final_data, uni_fea, fea_distribution, change_info


def align_feature_spaces(batches, uni_fea_batches):
    final_uni_fea = []
    for fea_list in uni_fea_batches:
        final_uni_fea = np.union1d(final_uni_fea, fea_list).tolist()

    aligned_batches = []
    for batch_idx, batch in enumerate(batches):
        current_fea = uni_fea_batches[batch_idx]

        labels = batch[:, -1:]

        features = batch[:, :-1]

        num_samples = batch.shape[0]
        aligned_features = np.full((num_samples, len(final_uni_fea)), np.nan)

        for i, fea_idx in enumerate(current_fea):
            col_idx = final_uni_fea.index(fea_idx)
            if i < features.shape[1]:
                aligned_features[:, col_idx] = features[:, i]

        aligned_batch = np.hstack((aligned_features, labels))
        aligned_batches.append(aligned_batch)

    return aligned_batches, final_uni_fea


def export_to_excel(aligned_batches, final_uni_fea, fea_distributions, change_info, output_file='aligned_batches.xlsx'):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        feature_columns = [f'feature_{idx}' for idx in final_uni_fea]
        columns = feature_columns + ['label']

        all_data = []
        for batch_idx, batch in enumerate(aligned_batches):
            batch_df = pd.DataFrame(batch, columns=columns)
            all_data.append(batch_df)

        result_df = pd.concat(all_data, ignore_index=True)
        result_df.to_excel(writer, sheet_name='All_Data', index=False)

        distribution_data = []
        for batch_idx, dist_dict in enumerate(fea_distributions):
            for feature_idx, dist_info in dist_dict.items():
                if feature_idx in final_uni_fea:
                    distribution_data.append({
                        'batch_id': batch_idx + 1,
                        'feature_id': feature_idx,
                        'distribution_type': dist_info[0],
                        'mean': dist_info[1],
                        'variance': dist_info[2]
                    })

        if distribution_data:
            dist_df = pd.DataFrame(distribution_data)
            dist_df.to_excel(writer, sheet_name='Feature_Distributions', index=False)

        if change_info:
            change_data = []
            distribution_change_details = []

            for batch_idx, info in enumerate(change_info):
                batch_num = batch_idx + 2
                change_dict = {
                    'batch_id': batch_num,
                    'change_type': info['type']
                }

                if 'silent_features' in info:
                    change_dict['silent_features'] = str(info['silent_features'])
                if 'new_features' in info:
                    change_dict['new_features'] = str(info['new_features'])

                if 'changed_features' in info:
                    change_dict['changed_distributions'] = str(info['changed_features'])
                elif 'changed_distributions' in info:
                    change_dict['changed_distributions'] = str(info['changed_distributions'])

                change_data.append(change_dict)

                if 'distribution_changes' in info and info['distribution_changes']:
                    for feat_idx, changes in info['distribution_changes'].items():
                        old_dist = changes['old']
                        new_dist = changes['new']
                        distribution_change_details.append({
                            'batch_id': batch_num,
                            'feature_id': feat_idx,
                            'old_distribution_type': old_dist[0],
                            'old_mean': old_dist[1],
                            'old_variance': old_dist[2],
                            'new_distribution_type': new_dist[0],
                            'new_mean': new_dist[1],
                            'new_variance': new_dist[2]
                        })

            change_df = pd.DataFrame(change_data)
            change_df.to_excel(writer, sheet_name='Change_Info', index=False)

            if distribution_change_details:
                dist_change_df = pd.DataFrame(distribution_change_details)
                dist_change_df.to_excel(writer, sheet_name='Distribution_Change_Details', index=False)

    return result_df

def check_label_balance(data):
    labels = data[:, -1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    counts_array = np.array(list(label_counts.values()))
    is_balanced = np.all(counts_array == counts_array[0])

    return is_balanced, label_counts


if __name__ == "__main__":
    num_samples = 2000
    ratio_change = 0.2
    ratio_silent = 0.1
    ratio_new = 0.3
    
    if num_samples % 2 != 0:
        num_samples += 1
        print(f"调整样本数为偶数: {num_samples}")

    change = [1, 2, 3, 1, 2, 3]  # example

    num_features_batch1 = 10
    data_batch1, fea_distribution1, uni_fea1 = generate_batch1(num_features_batch1, num_samples)

    batches = [data_batch1]
    uni_fea_batches = [uni_fea1]
    fea_distributions = [fea_distribution1]
    change_info_list = []

    current_fea = uni_fea1
    current_distribution = fea_distribution1

    for i, change_type in enumerate(change):
        data_batch, uni_fea, fea_distribution, change_info = generate_batch(
            change_type,
            num_samples,
            current_fea,
            current_distribution.copy(),
            ratio_change,
            ratio_silent,
            ratio_new
        )

        current_fea = uni_fea
        current_distribution = fea_distribution

        batches.append(data_batch)
        uni_fea_batches.append(uni_fea)
        fea_distributions.append(fea_distribution)
        change_info_list.append(change_info)

        print(f"已生成 Batch {i + 2}，变化类型: {change_type}")

    for i, batch in enumerate(batches):
        is_balanced, label_counts = check_label_balance(batch)
        print(f"Batch {i + 1} 标签平衡情况: {is_balanced}")
        print(f"Batch {i + 1} 标签分布: {label_counts}")

    aligned_batches, final_uni_fea = align_feature_spaces(batches, uni_fea_batches)

    result_df = export_to_excel(aligned_batches, final_uni_fea, fea_distributions, change_info_list,
                                'normal_scale/Change123.xlsx')

    for i, (batch, fea) in enumerate(zip(batches, uni_fea_batches)):
        print(f"Batch {i + 1} 原始特征数量: {len(fea)}, 原始数据形状: {batch.shape}")

    for i, batch in enumerate(aligned_batches):
        print(f"Batch {i + 1} 对齐后数据形状: {batch.shape}")

    print(f"最终特征总数: {len(final_uni_fea)}")
    print(f"最终特征索引: {final_uni_fea}")

    for batch_idx, dist_dict in enumerate(fea_distributions):
        print(f"\nBatch {batch_idx + 1} 特征分布情况:")
        for feat_idx, dist_info in dist_dict.items():
            if feat_idx in final_uni_fea:
                dist_type, mean, variance = dist_info
                print(f"  Feature {feat_idx}: 分布类型={dist_type}, 均值={mean}, 方差={variance}")

    for batch_idx, info in enumerate(change_info_list):
        print(f"\nBatch {batch_idx + 2} 特征变化情况:")

        change_type = info['type']
        if change_type == 'distribution_change':
            change_type_cn = "分布变化"
        elif change_type == 'space_change':
            change_type_cn = "特征空间变化"
        elif change_type == 'space_distribution_change':
            change_type_cn = "特征空间和分布同时变化"
        elif change_type == 'no_change':
            change_type_cn = "特征空间和分布均不变化"

        print(f"  变化类型: {change_type_cn}")

        if 'silent_features' in info:
            print(f"  静默特征: {info['silent_features']}")
        if 'new_features' in info:
            print(f"  新增特征: {info['new_features']}")
        if 'changed_features' in info:
            print(f"  分布变化特征: {info['changed_features']}")
        if 'changed_distributions' in info:
            print(f"  分布变化特征: {info['changed_distributions']}")

        if 'distribution_changes' in info and info['distribution_changes']:
            print("  详细分布变化信息:")
            for feat_idx, changes in info['distribution_changes'].items():
                old_dist = changes['old']
                new_dist = changes['new']
                print(f"    特征 {feat_idx}: 从 {old_dist} 变为 {new_dist}")