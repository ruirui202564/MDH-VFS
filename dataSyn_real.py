import numpy as np
import pandas as pd
from scipy.io import loadmat
import random

def generate_batch1(X, y, num_samples=2000, feature_ratio=0.7, start_idx=0):
    total_features = X.shape[1]
    num_used_features = int(total_features * feature_ratio)

    all_features = list(range(total_features))
    used_features = sorted(random.sample(all_features, num_used_features))
    remaining_features = sorted(list(set(all_features) - set(used_features)))

    end_idx = start_idx + num_samples
    X_batch = X[start_idx:end_idx, :][:, used_features]
    y_batch = y[start_idx:end_idx]

    feature_std = {}
    for i, feat_idx in enumerate(used_features):
        feature_std[feat_idx] = np.std(X_batch[:, i])

    data_batch = np.hstack((X_batch, y_batch.reshape(-1, 1)))

    return data_batch, used_features, remaining_features, feature_std


def change_distribution(X, y, used_features, feature_std, ratio_change=0.2, k=1.0,
                        start_idx=2000, num_samples=2000, changed_features_prev=None):
    end_idx = start_idx + num_samples
    num_change = max(1, int(np.round(len(used_features) * ratio_change)))
    num_change = min(num_change, len(used_features))

    if changed_features_prev is None:
        changed_features_prev = {}

    changed_features = random.sample(used_features, num_change)

    X_batch = X[start_idx:end_idx, :][:, used_features].copy()
    y_batch = y[start_idx:end_idx]

    new_feature_std = {}
    for i, feat_idx in enumerate(used_features):
        new_feature_std[feat_idx] = np.std(X_batch[:, i])

    distribution_changes = {}
    for feat_idx in used_features:
        feat_pos = used_features.index(feat_idx)

        if feat_idx in changed_features:
            if feat_idx in changed_features_prev:
                k_use = changed_features_prev[feat_idx]
            else:
                k_use = k

            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma

            distribution_changes[feat_idx] = k_use
        elif feat_idx in changed_features_prev:
            k_use = changed_features_prev[feat_idx]
            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma

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
    end_idx = start_idx + num_samples

    num_silent = max(1, int(np.round(len(used_features) * ratio_silent)))
    num_silent = min(num_silent, len(used_features) - 1) 
    silent_features = random.sample(used_features, num_silent)

    num_new = max(1, int(np.round(len(used_features) * ratio_new)))
    num_new = min(num_new, len(remaining_features))
    new_features = random.sample(remaining_features, num_new) if num_new > 0 else []

    active_features = [f for f in used_features if f not in silent_features]
    updated_used_features = sorted(active_features + new_features)
    updated_remaining = [f for f in remaining_features if f not in new_features]

    X_batch = X[start_idx:end_idx, :][:, updated_used_features]
    y_batch = y[start_idx:end_idx]

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
    end_idx = start_idx + num_samples

    if changed_features_prev is None:
        changed_features_prev = {}

    num_silent = max(1, int(np.round(len(used_features) * ratio_silent)))
    num_silent = min(num_silent, len(used_features) - 1)
    silent_features = random.sample(used_features, num_silent)

    num_new = len(remaining_features)
    new_features = remaining_features.copy()

    active_features = [f for f in used_features if f not in silent_features]
    updated_used_features = sorted(active_features + new_features)

    num_change = max(1, int(np.round(len(active_features) * ratio_change)))
    num_change = min(num_change, len(active_features))
    changed_features = random.sample(active_features, num_change)

    X_batch = X[start_idx:end_idx, :][:, updated_used_features].copy()
    y_batch = y[start_idx:end_idx]

    new_feature_std = {}
    for i, feat_idx in enumerate(updated_used_features):
        new_feature_std[feat_idx] = np.std(X_batch[:, i])

    distribution_changes = {}
    for feat_idx in updated_used_features:
        if feat_idx not in active_features:
            continue

        feat_pos = updated_used_features.index(feat_idx)

        if feat_idx in changed_features:
            if feat_idx in changed_features_prev:
                k_use = changed_features_prev[feat_idx]
            else:
                k_use = k

            sigma = feature_std.get(feat_idx, np.std(X_batch[:, feat_pos]))
            X_batch[:, feat_pos] = X_batch[:, feat_pos] + k_use * sigma
            distribution_changes[feat_idx] = k_use
        elif feat_idx in changed_features_prev:
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
    aligned_batches = []

    for batch, features in zip(batches, feature_lists):
        labels = batch[:, -1:]
        data = batch[:, :-1]

        num_samples = batch.shape[0]
        aligned_data = np.full((num_samples, total_features), np.nan)

        for i, feat_idx in enumerate(features):
            aligned_data[:, feat_idx] = data[:, i]

        aligned_batch = np.hstack((aligned_data, labels))
        aligned_batches.append(aligned_batch)

    return aligned_batches


def export_to_excel(aligned_batches, total_features, change_info_list, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        feature_columns = [f'feature_{i}' for i in range(total_features)]
        columns = feature_columns + ['label']

        all_data = []
        for batch_idx, batch in enumerate(aligned_batches):
            batch_df = pd.DataFrame(batch, columns=columns)
            all_data.append(batch_df)

        result_df = pd.concat(all_data, ignore_index=True)
        result_df.to_excel(writer, sheet_name='All_Data', index=False)

        change_data = []
        for batch_idx, info in enumerate(change_info_list):
            batch_num = batch_idx + 2
            change_dict = {
                'batch_id': batch_num,
                'change_type': info['type']
            }
            
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

        distribution_change_details = []
        for batch_idx, info in enumerate(change_info_list):
            batch_num = batch_idx + 2
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
    # first load real-world dataset, X and y
    num_samples = 2000
    k = 5.0

    total_samples, total_features = X.shape

    # check the number of samples
    required_samples =  num_samples * 4
    if total_samples < required_samples:
        print(f"warning: there only {total_samples} samples in the dataset，which is less than the required {required_samples}")

    # generating batch1
    print("\n generating Batch 1...")
    batch1, used_features1, remaining_features1, feature_std1 = generate_batch1(
        X, y, num_samples=num_samples, feature_ratio=0.7, start_idx=0
    )
    print(f"Batch 1: {len(used_features1)} features, {len(remaining_features1)} features are left")

    batches = [batch1]
    feature_lists = [used_features1]
    change_info_list = []

    # generating batch2
    print("\n generating Batch 2...")
    batch2, used_features2, feature_std2, change_info2, changed_dist2 = change_distribution(
        X, y, used_features1, feature_std1,
        ratio_change=0.2, k=k, start_idx=2000, num_samples=num_samples
    )
    batches.append(batch2)
    feature_lists.append(used_features2)
    change_info_list.append(change_info2)
    print(f"Batch 2: {len(change_info2['changed_features'])} features suffer from distribution change")

    # generate batch3
    print("\n generating Batch 3...")
    batch3, used_features3, remaining_features3, feature_std3, change_info3 = change_space(
        X, y, used_features2, remaining_features1, feature_std2,
        ratio_silent=0.1, ratio_new=0.2, start_idx=4000, num_samples=num_samples
    )
    batches.append(batch3)
    feature_lists.append(used_features3)
    change_info_list.append(change_info3)
    print(f"Batch 3: remove {len(change_info3['silent_features'])} features, add {len(change_info3['new_features'])} features")

    # generating batch4
    print("\n generating Batch 4...")
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
        f"Batch 4: remove {len(change_info4['silent_features'])} features, add {len(change_info4['new_features'])} features, {len(change_info4['changed_features'])} features suffer from distribution change")

    # align the feature space
    print("\n align the feature space...")
    aligned_batches = align_feature_spaces(batches, feature_lists, total_features)

    # output
    output_file = f'realData_new/electricity_{int(k)}_.xlsx'
    print(f"\n output to {output_file}...")
    export_to_excel(aligned_batches, total_features, change_info_list, output_file)

    print("\n Complete!")
    print(f"The number of total features: {total_features}")
    for i, features in enumerate(feature_lists):

        print(f"The number of features in Batch {i + 1}: {len(features)}")