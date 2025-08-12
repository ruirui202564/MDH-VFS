import math
import pandas as pd
from collections import defaultdict
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import os
import openpyxl
import numpy as np

def read_excel_to_driftposition(file_path, sheet_name='Distribution_Change_Details'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    drift_dict = defaultdict(list)

    for feature_id in df['feature_id'].unique():
        batch_ids = df[df['feature_id'] == feature_id]['batch_id'].tolist()

        drift_positions = [((batch_id - 1) * 2000 + 1) for batch_id in batch_ids]

        drift_dict[str(feature_id)] = sorted(drift_positions)

    return dict(drift_dict)

def read_excel_to_detected_drifts(excel_file, sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    drift_dict = defaultdict(list)

    for _, row in df.iterrows():
        feature = str(row['drift_feature'])
        position = int(row['drift_position'])
        drift_dict[feature].append(position)

    for feature in drift_dict:
        drift_dict[feature] = sorted(list(set(drift_dict[feature])))

    detected_drifts = [dict(drift_dict)]
    return detected_drifts

def evaluation(detected_drifts, detectiondelay, driftposition, fullsize):
    '''
    :param detected_drifts: list of drift detection results, each is a dict {feature: [drift points]}
    :param detectiondelay: list of tolerable delays (int)
    :param driftposition: dict {feature: [ground truth drift points]}
    :param fullsize: total number of data points
    '''
    _recall = []
    _precision = []
    _F1 = []
    _MCC = []
    _muD = []

    for detector in detected_drifts:
        recall_l = []
        precision_l = []
        F1_l = []
        MCC_l = []
        muD_l = []

        for d_delay in detectiondelay:
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            delay_sum = 0

            for feature, real_drifts in driftposition.items():
                detected = detector.get(feature, [])
                start = 0

                for end in real_drifts:
                    flag = 0
                    for val in detected:
                        if start <= val < end:
                            FP += 1
                        if flag == 0 and end <= val < (end + d_delay):
                            TP += 1
                            flag = 1
                            delay_sum += val - end
                        elif flag == 1 and end <= val < (end + d_delay):
                            FP += 1

                    if flag == 0:
                        FN += 1
                    start = end + d_delay

                for val in detected:
                    if start <= val < fullsize:
                        FP += 1

            recall = TP / (TP + FN) if (TP + FN) != 0 else 0

            if TP == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)

            F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            TN = fullsize - FP - TP- FN

            denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            if denominator == 0:
                MCC = 0
            else:
                MCC = (TP * TN - FP * FN) / denominator

            if TP >0:
                muD = delay_sum / TP
            else:
                muD = np.nan

            recall_l.append(recall)
            precision_l.append(precision)
            F1_l.append(F1)
            MCC_l.append(MCC)
            muD_l.append(muD)

        _recall.append(recall_l)
        _precision.append(precision_l)
        _F1.append(F1_l)
        _MCC.append(MCC_l)
        _muD.append(muD_l)

    return _recall, _precision, _F1, _MCC, _muD, TP, TN, FN, FP

if __name__ == "__main__":
    dataset = ['Change123']
    detector_all = ['HD']
    driftposition_path = '../SyntheticData/normal/'
    detected_path = 'result/normal/'
    res_path = 'result/normal/'

    referNum_test = [4]
    referValue_test = ['center']
    for referNum in referNum_test:
        for referValue in referValue_test:
            breakpoint = f'{referNum}_{referValue}'
            for detector in detector_all:
                for data in dataset:
                    position_path = f"{driftposition_path}{data}.xlsx"
                    driftposition = read_excel_to_driftposition(position_path, sheet_name='Distribution_Change_Details')
                    detected_drifts_path = f"{detected_path}{data}_{detector}_{referNum}_{referValue}.xlsx"
                    detected_results = pd.ExcelFile(detected_drifts_path)
                    if (breakpoint == '3_log' or breakpoint == '3_linear') and detector == 'HD' and data == 'Change312312':
                        detected_drifts_path2 = f"{detected_path}{data}_{detector} old.xlsx"
                        detected_results2 = pd.ExcelFile(detected_drifts_path2)
                        all_sheet_names = list(detected_results.sheet_names) + list(detected_results2.sheet_names)
                        method_all = [name for name in all_sheet_names if breakpoint in name]
                    else:
                        method_all = [name for name in detected_results.sheet_names if
                                      name != 'runtime' and breakpoint in name]

                    for i, method in enumerate(method_all):
                        print(f"当前的detector: {detector}", f"当前的数据集: {data}", f"当前的method: {method}")
                        result_path = f"{res_path}res_detect_normal_{detector}_{breakpoint}_split.xlsx"
                        if os.path.exists(result_path):
                            existing_result = pd.ExcelFile(result_path)
                            if data in list(existing_result.sheet_names):
                                df = pd.read_excel(result_path, sheet_name=data)
                                if len(df) > 0 and method in df['method'].values:
                                    print(f"跳过已存在的method: {method}")
                                    continue

                        detected_drifts_path = f"{detected_path}{data}_{detector}_{referNum}_{referValue}.xlsx"
                        method_found = False

                        with pd.ExcelFile(detected_drifts_path) as excel:
                            if method in excel.sheet_names:
                                detected_drifts = read_excel_to_detected_drifts(detected_drifts_path, sheet_name=method)
                                method_found = True
                        if not method_found and (breakpoint == '3_log' or breakpoint == '3_linear') and\
                                detector == 'HD' and data == 'Change312312':
                            detected_drifts_path_old = f"{detected_path}{data}_{detector} old.xlsx"
                            if os.path.exists(detected_drifts_path_old):
                                with pd.ExcelFile(detected_drifts_path_old) as excel_old:
                                    if method in excel_old.sheet_names:
                                        detected_drifts = read_excel_to_detected_drifts(detected_drifts_path_old,
                                                                                        sheet_name=method)
                                        method_found = True

                        if not method_found:
                            print(f"Sheet '{method}' 不存在于任何文件中")
                            continue

                        _recall, _precision, _F1, _MCC, _muD, TP, TN, FN, FP = evaluation(detected_drifts, [40],
                                                                                          driftposition, 14000)

                        print('recall:', _recall)
                        print('precision:', _precision)
                        print('F1:', _F1)
                        print('MCC:', _MCC)
                        print('muD:', _muD)
                        print('TP:', TP)
                        print('FN:', FN)
                        print('FP:', FP)
                        print('TN:', TN)

                        method_parts = method.split('_')
                        if len(method_parts) >= 6:
                            win_val1 = method_parts[0]
                            win_val2 = method_parts[1]
                            referNum_val = method_parts[2]
                            referValue_val = method_parts[3]
                            warn_val = method_parts[4]
                            upThres_val = method_parts[5]
                            k_val = method_parts[6]
                        else:
                            win_val1 = method_parts[0] if len(method_parts) > 0 else ''
                            win_val2 = method_parts[1] if len(method_parts) > 1 else ''
                            referNum_val = method_parts[2] if len(method_parts) > 2 else ''
                            referValue_val = method_parts[3] if len(method_parts) > 3 else ''
                            warn_val = method_parts[4] if len(method_parts) > 4 else ''
                            upThres_val = method_parts[5] if len(method_parts) > 5 else ''
                            k_val = method_parts[6] if len(method_parts) > 6 else ''

                        new_row = {
                            "method": method,
                            "referNum": referNum_val,
                            "referValue": referValue_val,
                            "warn": warn_val,
                            "upThres": upThres_val,
                            "k": k_val,
                            "w1": win_val1,
                            "w2": win_val2,
                            "Recall": _recall[0][0],
                            "Precision": _precision[0][0],
                            "F1-score": _F1[0][0],
                            "MCC": _MCC[0][0],
                            "muD": _muD[0][0],
                            "TP": TP,
                            "TN": TN,
                            "FN": FN,
                            "FP": FP
                        }

                        sheet_name = data
                        if os.path.exists(result_path):
                            try:
                                existing_df = pd.read_excel(result_path, sheet_name=sheet_name)
                            except ValueError:
                                existing_df = pd.DataFrame()

                            new_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)

                            with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='replace') as writer:
                                new_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        else:
                            new_df = pd.DataFrame([new_row])
                            with pd.ExcelWriter(result_path) as writer:
                                new_df.to_excel(writer, sheet_name=sheet_name, index=False)