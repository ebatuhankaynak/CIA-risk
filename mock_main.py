from data_handler import MockDataHandler
import numpy as np

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
]

features_per_cid = dh.create_features("Jsoup", requested_features)
y = dh.create_labels("Jsoup")

def flatten_features(features_dict, y):
    flattened_lists = []
    for cid in features_dict:
        flattened_lists.append([])
        for feat_key in features_dict[cid]:
            flattened_lists[-1].extend(features_dict[cid][feat_key])

    
    max_length = max(len(lst) for lst in flattened_lists)
    filtered_flattened_lists = []
    filtered_y = []

    for i, lst in enumerate(flattened_lists):
        if len(lst) == max_length:
            filtered_flattened_lists.append(lst)
            filtered_y.append(y[i])

    x = np.array(filtered_flattened_lists)
    y = np.array(filtered_y)

    means = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), means, x)

    return x, y

x, y = flatten_features(features_per_cid, y)

print(x)
print(y)


import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.5, random_state=42)

ocsvm = OneClassSVM(gamma='auto')
ocsvm.fit(x_train)

decision_function = ocsvm.decision_function(x_val)
confidence_scores = 1 / (1 + np.exp(-decision_function))

threshold = 0.5
y_pred = (decision_function > threshold).astype(int)

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Normal', 'BIC']))

roc_auc = roc_auc_score(y_val, confidence_scores)
print("ROC AUC Score:", roc_auc)
