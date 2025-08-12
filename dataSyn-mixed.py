import numpy as np
import random
import pandas as pd

def generate_batch1(num_features, num_samples, distributions=None):
    data = np.zeros((num_samples, num_features))
    fea_distribution = {}

    if distributions is None:
        distributions = random.choices(['normal_5', 'exponential', 'uniform', 'beta', 'gamma'], k=num_features)
    elif len(distributions) < num_features:
        additional = random.choices(['normal_5', 'exponential', 'uniform', 'beta', 'gamma'], k=num_features - len(distributions))
        distributions.extend(additional)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, num_features))

    for i in range(num_features):
        dist_type = distributions[i]
        if dist_type == 'normal_5':
            mean = 0
            variance = 1
            extra_data[:, i] = np.random.normal(mean, variance, extra_samples)
            fea_distribution[i] = ('normal_5', mean, variance)
        elif dist_type == 'exponential':
            scale = 1
            extra_data[:, i] = np.random.exponential(scale, extra_samples)
            fea_distribution[i] = ('exponential', scale)
        elif dist_type == 'uniform':
            low = 0
            high = 1
            extra_data[:, i] = np.random.uniform(low, high, extra_samples)
            fea_distribution[i] = ('uniform', low, high)
        elif dist_type == 'beta':
            shape = 1
            scale = 2
            extra_data[:, i] = np.random.beta(shape, scale, extra_samples)
            fea_distribution[i] = ('beta', shape, scale)
        elif dist_type == 'gamma':
            shape = 1
            scale = 2
            extra_data[:, i] = np.random.gamma(shape, scale, extra_samples)
            fea_distribution[i] = ('gamma', shape, scale)

    means_list = []
    for i in range(num_features):
        dist_info = fea_distribution[i]
        if dist_info[0] == 'normal_5':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'exponential':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'uniform':
            means_list.append((dist_info[1] + dist_info[2]) / 2)
        elif dist_info[0] == 'beta':
            alpha, beta = dist_info[1], dist_info[2]
            means_list.append(alpha / (alpha + beta))
        elif dist_info[0] == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            means_list.append(shape * scale)

    threshold = np.sum(means_list)
    f = np.sum(extra_data[:, :], axis=1)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return generate_batch1(num_features, num_samples, distributions)

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
        dist_info = fea_distribution[uni_fea_old[i]]
        dist_type = dist_info[0]
        if dist_type == 'normal_5':
            mean, variance = dist_info[1], dist_info[2]
            extra_data[:, i] = np.random.normal(mean, variance, extra_samples)
        elif dist_type == 'exponential':
            scale = dist_info[1]
            extra_data[:, i] = np.random.exponential(scale, extra_samples)
        elif dist_type == 'uniform':
            low, high = dist_info[1], dist_info[2]
            extra_data[:, i] = np.random.uniform(low, high, extra_samples)
        elif dist_type == 'beta':
            shape, scale = dist_info[1], dist_info[2]
            extra_data[:, i] = np.random.beta(shape, scale, extra_samples)
            fea_distribution[i] = ('beta', shape, scale)
        elif dist_type == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            extra_data[:, i] = np.random.gamma(shape, scale, extra_samples)
            fea_distribution[i] = ('gamma', shape, scale)

    f = np.sum(extra_data, axis=1)
    means_list = []
    for idx in uni_fea_old:
        dist_info = fea_distribution[idx]
        if dist_info[0] == 'normal_5':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'exponential':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'uniform':
            means_list.append((dist_info[1] + dist_info[2]) / 2)
        elif dist_info[0] == 'beta':
            alpha, beta = dist_info[1], dist_info[2]
            means_list.append(alpha / (alpha + beta))
        elif dist_info[0] == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            means_list.append(shape * scale)

    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)
    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return no_change(num_samples, uni_fea_old, fea_distribution)

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
        feat_idx = uni_fea_old[i]
        dist_info = fea_distribution[feat_idx]
        dist_type = dist_info[0]

        if feat_idx in change:
            if dist_type == 'normal_5':
                old_mean, old_variance = dist_info[1], dist_info[2]
                mean_new = random.uniform(-5, 5)
                variance_new = random.uniform(1, 3)
                new_mean = old_mean + mean_new
                new_variance = old_variance + variance_new

                distribution_changes[feat_idx] = {
                    'old': (dist_type, old_mean, old_variance),
                    'new': (dist_type, new_mean, new_variance)
                }
                fea_distribution[feat_idx] = ('normal_5', new_mean, new_variance)
                extra_data[:, i] = np.random.normal(new_mean, new_variance, extra_samples)
            elif dist_type == 'exponential':
                old_scale = dist_info[1]
                scale_new = old_scale + random.uniform(0.5, 2)

                distribution_changes[feat_idx] = {
                    'old': (dist_type, old_scale),
                    'new': (dist_type, scale_new)
                }
                fea_distribution[feat_idx] = ('exponential', scale_new)
                extra_data[:, i] = np.random.exponential(scale_new, extra_samples)
            elif dist_type == 'uniform':
                old_low, old_high = dist_info[1], dist_info[2]
                new_low = old_low - random.uniform(0, 5)
                new_high = old_high + random.uniform(0, 5)
                distribution_changes[feat_idx] = {
                    'old': (dist_type, old_low, old_high),
                    'new': (dist_type, new_low, new_high)
                }
                fea_distribution[feat_idx] = ('uniform', new_low, new_high)
                extra_data[:, i] = np.random.uniform(new_low, new_high, extra_samples)
            elif dist_type == 'beta':
                old_low, old_high = dist_info[1], dist_info[2]
                new_low = old_low + random.uniform(0, 5)
                new_high = old_high + random.uniform(0, 5)
                distribution_changes[feat_idx] = {
                    'old': (dist_type, old_low, old_high),
                    'new': (dist_type, new_low, new_high)
                }
                fea_distribution[feat_idx] = ('beta', new_low, new_high)
                extra_data[:, i] = np.random.beta(new_low, new_high, extra_samples)
            elif dist_type == 'gamma':
                old_low, old_high = dist_info[1], dist_info[2]
                new_low = old_low + random.uniform(0, 5)
                new_high = old_high + random.uniform(0, 5)
                distribution_changes[feat_idx] = {
                    'old': (dist_type, old_low, old_high),
                    'new': (dist_type, new_low, new_high)
                }
                fea_distribution[feat_idx] = ('gamma', new_low, new_high)
                extra_data[:, i] = np.random.gamma(new_low, new_high, extra_samples)

            if random.random() < 0.3:
                old_dist_type = dist_type
                new_dist_types = [d for d in ['normal_5', 'exponential', 'uniform', 'beta', 'gamma'] if d != old_dist_type]
                new_dist_type = random.choice(new_dist_types)

                if new_dist_type == 'normal_5':
                    new_mean = random.uniform(-2, 2)
                    new_variance = random.uniform(0.5, 2)
                    distribution_changes[feat_idx] = {
                        'old': dist_info,
                        'new': (new_dist_type, new_mean, new_variance)
                    }
                    fea_distribution[feat_idx] = (new_dist_type, new_mean, new_variance)
                    extra_data[:, i] = np.random.normal(new_mean, new_variance, extra_samples)
                elif new_dist_type == 'exponential':
                    new_scale = random.uniform(0.5, 2)
                    distribution_changes[feat_idx] = {
                        'old': dist_info,
                        'new': (new_dist_type, new_scale)
                    }
                    fea_distribution[feat_idx] = (new_dist_type, new_scale)
                    extra_data[:, i] = np.random.exponential(new_scale, extra_samples)
                elif new_dist_type == 'uniform':
                    new_low = random.uniform(-5, 0)
                    new_high = random.uniform(1, 5)
                    distribution_changes[feat_idx] = {
                        'old': dist_info,
                        'new': (new_dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                    extra_data[:, i] = np.random.uniform(new_low, new_high, extra_samples)
                elif new_dist_type == 'beta':
                    new_low = random.uniform(1, 5)
                    new_high = random.uniform(1, 5)
                    distribution_changes[feat_idx] = {
                        'old': dist_info,
                        'new': (new_dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                    extra_data[:, i] = np.random.beta(new_low, new_high, extra_samples)
                elif new_dist_type == 'gamma':
                    new_low = random.uniform(1, 5)
                    new_high = random.uniform(1, 5)
                    distribution_changes[feat_idx] = {
                        'old': dist_info,
                        'new': (new_dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                    extra_data[:, i] = np.random.gamma(new_low, new_high, extra_samples)
        else:
            if dist_type == 'normal_5':
                mean, variance = dist_info[1], dist_info[2]
                extra_data[:, i] = np.random.normal(mean, variance, extra_samples)
            elif dist_type == 'exponential':
                scale = dist_info[1]
                extra_data[:, i] = np.random.exponential(scale, extra_samples)
            elif dist_type == 'uniform':
                low, high = dist_info[1], dist_info[2]
                extra_data[:, i] = np.random.uniform(low, high, extra_samples)
            elif dist_type == 'beta':
                low, high = dist_info[1], dist_info[2]
                extra_data[:, i] = np.random.beta(low, high, extra_samples)
            elif dist_type == 'gamma':
                low, high = dist_info[1], dist_info[2]
                extra_data[:, i] = np.random.gamma(low, high, extra_samples)

    f = np.sum(extra_data, axis=1)
    means_list = []
    for idx in uni_fea_old:
        dist_info = fea_distribution[idx]
        if dist_info[0] == 'normal_5':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'exponential':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'uniform':
            means_list.append((dist_info[1] + dist_info[2]) / 2)
        elif dist_info[0] == 'beta':
            alpha, beta = dist_info[1], dist_info[2]
            means_list.append(alpha / (alpha + beta))
        elif dist_info[0] == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            means_list.append(shape * scale)

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

def change_space(num_samples, uni_fea_old, ratio_silent, ratio_new, fea_distribution):
    num_fea_old = len(uni_fea_old)
    uni_fea_old_list = uni_fea_old.tolist() if isinstance(uni_fea_old, np.ndarray) else list(uni_fea_old)
    num_silent = int(np.round(num_fea_old * ratio_silent))
    silent = random.sample(uni_fea_old_list, num_silent)
    num_new = int(np.round(num_fea_old * ratio_new))
    last_fea = max(uni_fea_old)
    new = [last_fea + i for i in range(1, num_new + 1)]
    new = np.array(new)
    active = np.setdiff1d(uni_fea_old, silent)
    uni_fea = np.union1d(uni_fea_old, new)

    extra_samples = num_samples * 2
    extra_data = np.zeros((extra_samples, len(uni_fea)))
    extra_data[:, silent] = np.nan

    for feat_idx in active:
        dist_info = fea_distribution[feat_idx]
        dist_type = dist_info[0]
        if dist_type == 'normal_5':
            mean, variance = dist_info[1], dist_info[2]
            extra_data[:, feat_idx] = np.random.normal(mean, variance, extra_samples)
        elif dist_type == 'exponential':
            scale = dist_info[1]
            extra_data[:, feat_idx] = np.random.exponential(scale, extra_samples)
        elif dist_type == 'uniform':
            low, high = dist_info[1], dist_info[2]
            extra_data[:, feat_idx] = np.random.uniform(low, high, extra_samples)
        elif dist_type == 'beta':
            low, high = dist_info[1], dist_info[2]
            extra_data[:, feat_idx] = np.random.beta(low, high, extra_samples)
        elif dist_type == 'gamma':
            low, high = dist_info[1], dist_info[2]
            extra_data[:, feat_idx] = np.random.gamma(low, high, extra_samples)

    for feat_idx in new:
        dist_type = random.choice(['normal_5', 'exponential', 'uniform', 'beta', 'gamma'])
        if dist_type == 'normal_5':
            mean = random.uniform(-2, 2)
            variance = random.uniform(0.5, 2)
            fea_distribution[feat_idx] = ('normal_5', mean, variance)
            extra_data[:, feat_idx] = np.random.normal(mean, variance, extra_samples)
        elif dist_type == 'exponential':
            scale = random.uniform(0.5, 2)
            fea_distribution[feat_idx] = ('exponential', scale)
            extra_data[:, feat_idx] = np.random.exponential(scale, extra_samples)
        elif dist_type == 'uniform':
            low = random.uniform(-5, 0)
            high = random.uniform(1, 5)
            fea_distribution[feat_idx] = ('uniform', low, high)
            extra_data[:, feat_idx] = np.random.uniform(low, high, extra_samples)
        elif dist_type == 'beta':
            low = random.uniform(1, 5)
            high = random.uniform(1, 5)
            fea_distribution[feat_idx] = ('beta', low, high)
            extra_data[:, feat_idx] = np.random.beta(low, high, extra_samples)
        elif dist_type == 'gamma':
            low = random.uniform(1, 5)
            high = random.uniform(1, 5)
            fea_distribution[feat_idx] = ('gamma', low, high)
            extra_data[:, feat_idx] = np.random.gamma(low, high, extra_samples)

    current = np.concatenate((active, new))
    f = np.nansum(extra_data[:, current], axis=1)

    means_list = []
    for idx in current:
        dist_info = fea_distribution[idx]
        if dist_info[0] == 'normal_5':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'exponential':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'uniform':
            means_list.append((dist_info[1] + dist_info[2]) / 2)
        elif dist_info[0] == 'beta':
            alpha, beta = dist_info[1], dist_info[2]
            means_list.append(alpha / (alpha + beta))
        elif dist_info[0] == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            means_list.append(shape * scale)

    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_space(num_samples, uni_fea_old, ratio_silent, ratio_new, fea_distribution)

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


def change_space_distribution(num_samples, uni_fea_old, ratio_silent, ratio_new, ratio_change, fea_distribution):
    num_fea_old = len(uni_fea_old)
    uni_fea_old_list = uni_fea_old.tolist() if isinstance(uni_fea_old, np.ndarray) else list(uni_fea_old)
    num_silent = int(np.round(num_fea_old * ratio_silent))
    silent = random.sample(uni_fea_old_list, num_silent)
    num_new = int(np.round(num_fea_old * ratio_new))
    last_fea = max(uni_fea_old)
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
        feat_idx = uni_fea[i]
        if feat_idx in new: 
            dist_type = random.choice(['normal_5', 'exponential', 'uniform', 'beta', 'gamma'])
            if dist_type == 'normal_5':
                mean = random.uniform(-2, 2)
                variance = random.uniform(0.5, 2)
                fea_distribution[feat_idx] = ('normal_5', mean, variance)
                extra_data[:, feat_idx] = np.random.normal(mean, variance, extra_samples)
            elif dist_type == 'exponential':
                scale = random.uniform(0.5, 2)
                fea_distribution[feat_idx] = ('exponential', scale)
                extra_data[:, feat_idx] = np.random.exponential(scale, extra_samples)
            elif dist_type == 'uniform':
                low = random.uniform(-5, 0)
                high = random.uniform(1, 5)
                fea_distribution[feat_idx] = ('uniform', low, high)
                extra_data[:, feat_idx] = np.random.uniform(low, high, extra_samples)
            elif dist_type == 'beta':
                low = random.uniform(1, 5)
                high = random.uniform(1, 5)
                fea_distribution[feat_idx] = ('beta', low, high)
                extra_data[:, feat_idx] = np.random.beta(low, high, extra_samples)
            elif dist_type == 'gamma':
                low = random.uniform(1, 5)
                high = random.uniform(1, 5)
                fea_distribution[feat_idx] = ('gamma', low, high)
                extra_data[:, feat_idx] = np.random.gamma(low, high, extra_samples)
        elif feat_idx in active:
            dist_info = fea_distribution[feat_idx]
            dist_type = dist_info[0]
            if feat_idx in change:
                if dist_type == 'normal_5':
                    old_mean, old_variance = dist_info[1], dist_info[2]
                    mean_new = random.uniform(-5, 5)
                    variance_new = random.uniform(1, 3)
                    new_mean = old_mean + mean_new
                    new_variance = old_variance + variance_new

                    distribution_changes[feat_idx] = {
                        'old': (dist_type, old_mean, old_variance),
                        'new': (dist_type, new_mean, new_variance)
                    }
                    fea_distribution[feat_idx] = ('normal_5', new_mean, new_variance)
                    extra_data[:, i] = np.random.normal(new_mean, new_variance, extra_samples)

                elif dist_type == 'exponential':
                    old_scale = dist_info[1]
                    scale_new = old_scale + random.uniform(0.5, 2)

                    distribution_changes[feat_idx] = {
                        'old': (dist_type, old_scale),
                        'new': (dist_type, scale_new)
                    }
                    fea_distribution[feat_idx] = ('exponential', scale_new)
                    extra_data[:, i] = np.random.exponential(scale_new, extra_samples)

                elif dist_type == 'uniform':
                    old_low, old_high = dist_info[1], dist_info[2]
                    new_low = old_low - random.uniform(0, 2)
                    new_high = old_high + random.uniform(0, 2)

                    distribution_changes[feat_idx] = {
                        'old': (dist_type, old_low, old_high),
                        'new': (dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = ('uniform', new_low, new_high)
                    extra_data[:, i] = np.random.uniform(new_low, new_high, extra_samples)
                elif dist_type == 'beta':
                    old_low, old_high = dist_info[1], dist_info[2]
                    new_low = old_low + random.uniform(0, 5)
                    new_high = old_high + random.uniform(0, 5)
                    distribution_changes[feat_idx] = {
                        'old': (dist_type, old_low, old_high),
                        'new': (dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = ('beta', new_low, new_high)
                    extra_data[:, i] = np.random.beta(new_low, new_high, extra_samples)
                elif dist_type == 'gamma':
                    old_low, old_high = dist_info[1], dist_info[2]
                    new_low = old_low + random.uniform(0, 5)
                    new_high = old_high + random.uniform(0, 5)
                    distribution_changes[feat_idx] = {
                        'old': (dist_type, old_low, old_high),
                        'new': (dist_type, new_low, new_high)
                    }
                    fea_distribution[feat_idx] = ('gamma', new_low, new_high)
                    extra_data[:, i] = np.random.gamma(new_low, new_high, extra_samples)

                if random.random() < 0.3:
                    old_dist_type = dist_type
                    new_dist_types = [d for d in ['normal_5', 'exponential', 'uniform'] if d != old_dist_type]
                    new_dist_type = random.choice(new_dist_types)

                    if new_dist_type == 'normal_5':
                        new_mean = random.uniform(-2, 2)
                        new_variance = random.uniform(0.5, 2)
                        distribution_changes[feat_idx] = {
                            'old': dist_info,
                            'new': (new_dist_type, new_mean, new_variance)
                        }
                        fea_distribution[feat_idx] = (new_dist_type, new_mean, new_variance)
                        extra_data[:, i] = np.random.normal(new_mean, new_variance, extra_samples)
                    elif new_dist_type == 'exponential':
                        new_scale = random.uniform(0.5, 2)
                        distribution_changes[feat_idx] = {
                            'old': dist_info,
                            'new': (new_dist_type, new_scale)
                        }
                        fea_distribution[feat_idx] = (new_dist_type, new_scale)
                        extra_data[:, i] = np.random.exponential(new_scale, extra_samples)

                    elif new_dist_type == 'uniform':
                        new_low = random.uniform(-5, 0)
                        new_high = random.uniform(1, 5)
                        distribution_changes[feat_idx] = {
                            'old': dist_info,
                            'new': (new_dist_type, new_low, new_high)
                        }
                        fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                        extra_data[:, i] = np.random.uniform(new_low, new_high, extra_samples)
                    elif new_dist_type == 'beta':
                        new_low = random.uniform(1, 5)
                        new_high = random.uniform(1, 5)
                        distribution_changes[feat_idx] = {
                            'old': dist_info,
                            'new': (new_dist_type, new_low, new_high)
                        }
                        fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                        extra_data[:, i] = np.random.beta(new_low, new_high, extra_samples)
                    elif new_dist_type == 'gamma':
                        new_low = random.uniform(1, 5)
                        new_high = random.uniform(1, 5)
                        distribution_changes[feat_idx] = {
                            'old': dist_info,
                            'new': (new_dist_type, new_low, new_high)
                        }
                        fea_distribution[feat_idx] = (new_dist_type, new_low, new_high)
                        extra_data[:, i] = np.random.gamma(new_low, new_high, extra_samples)
            else:
                if dist_type == 'normal_5':
                    mean, variance = dist_info[1], dist_info[2]
                    extra_data[:, i] = np.random.normal(mean, variance, extra_samples)
                elif dist_type == 'exponential':
                    scale = dist_info[1]
                    extra_data[:, i] = np.random.exponential(scale, extra_samples)
                elif dist_type == 'uniform':
                    low, high = dist_info[1], dist_info[2]
                    extra_data[:, i] = np.random.uniform(low, high, extra_samples)
                elif dist_type == 'beta':
                    low, high = dist_info[1], dist_info[2]
                    extra_data[:, i] = np.random.beta(low, high, extra_samples)
                elif dist_type == 'gamma':
                    low, high = dist_info[1], dist_info[2]
                    extra_data[:, i] = np.random.gamma(low, high, extra_samples)

    current = np.concatenate((active, new))
    f = np.nansum(extra_data[:, current], axis=1)

    means_list = []
    for idx in current:
        dist_info = fea_distribution[idx]
        if dist_info[0] == 'normal_5':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'exponential':
            means_list.append(dist_info[1])
        elif dist_info[0] == 'uniform':
            means_list.append((dist_info[1] + dist_info[2]) / 2)
        elif dist_info[0] == 'beta':
            alpha, beta = dist_info[1], dist_info[2]
            means_list.append(alpha / (alpha + beta))
        elif dist_info[0] == 'gamma':
            shape, scale = dist_info[1], dist_info[2]
            means_list.append(shape * scale)

    threshold = np.sum(means_list)
    extra_labels = (f < threshold).astype(int)

    idx_label_0 = np.where(extra_labels == 0)[0]
    idx_label_1 = np.where(extra_labels == 1)[0]

    if len(idx_label_0) < num_samples // 2 or len(idx_label_1) < num_samples // 2:
        return change_space(num_samples, uni_fea_old, ratio_silent, ratio_new, fea_distribution)

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
                    if dist_info[0] in ['normal_5', 'uniform', 'beta', 'gamma']:
                        distribution_data.append({
                            'batch_id': batch_idx + 1,
                            'feature_id': feature_idx,
                            'distribution_type': dist_info[0],
                            'param1': dist_info[1],
                            'param2': dist_info[2]
                        })
                    elif dist_info[0] == 'exponential':
                        distribution_data.append({
                            'batch_id': batch_idx + 1,
                            'feature_id': feature_idx,
                            'distribution_type': dist_info[0],
                            'param1': dist_info[1]
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
                        if old_dist[0] == 'exponential':
                            if new_dist[0] == 'exponential':
                                distribution_change_details.append({
                                    'batch_id': batch_num,
                                    'feature_id': feat_idx,
                                    'old_distribution_type': old_dist[0],
                                    'old_param1': old_dist[1],
                                    'new_distribution_type': new_dist[0],
                                    'new_param1': new_dist[1]
                                })
                            else:
                                distribution_change_details.append({
                                    'batch_id': batch_num,
                                    'feature_id': feat_idx,
                                    'old_distribution_type': old_dist[0],
                                    'old_param1': old_dist[1],
                                    'new_distribution_type': new_dist[0],
                                    'new_param1': new_dist[1],
                                    'new_param2': new_dist[2]
                                })
                        else:
                            if new_dist[0] == 'exponential':
                                distribution_change_details.append({
                                    'batch_id': batch_num,
                                    'feature_id': feat_idx,
                                    'old_distribution_type': old_dist[0],
                                    'old_param1': old_dist[1],
                                    'old_param2': old_dist[2],
                                    'new_distribution_type': new_dist[0],
                                    'new_param1': new_dist[1]
                                })
                            else:
                                distribution_change_details.append({
                                    'batch_id': batch_num,
                                    'feature_id': feat_idx,
                                    'old_distribution_type': old_dist[0],
                                    'old_param1': old_dist[1],
                                    'old_param2': old_dist[2],
                                    'new_distribution_type': new_dist[0],
                                    'new_param1': new_dist[1],
                                    'new_param2': new_dist[2]
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

    change = [2, 3, 1, 2, 3, 1]   # example

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
                                'mixed_new/Change231231.xlsx')

    for i, (batch, fea) in enumerate(zip(batches, uni_fea_batches)):
        print(f"Batch {i + 1} 原始特征数量: {len(fea)}, 原始数据形状: {batch.shape}")

    for i, batch in enumerate(aligned_batches):
        print(f"Batch {i + 1} 对齐后数据形状: {batch.shape}")

    print(f"最终特征总数: {len(final_uni_fea)}")
    print(f"最终特征索引: {final_uni_fea}")

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