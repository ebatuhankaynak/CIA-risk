from data_handler import MockDataHandler
import numpy as np
import matplotlib.pyplot as plt

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
]

project_name = "Cli"

y = dh.create_labels(project_name)
if False:
    y = dh.create_labels(project_name)
    features_per_cid = dh.create_features_from_cc(project_name, requested_features)
    

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

    np.save(f"{project_name}_x", x)
    np.save(f"{project_name}_y", y)
else:
    x = np.load(f"{project_name}_x.npy")
    y = np.load(f"{project_name}_y.npy")


from sklearn.ensemble import IsolationForest
import numpy as np

iso_forest = IsolationForest(contamination=sum(y)/len(y))
iso_forest.fit(x)
raw_scores = iso_forest.decision_function(x)
normalized_scores = 1 - 1 / (1 + np.exp(-raw_scores))
print(max(raw_scores), min(raw_scores))
print(max(normalized_scores), min(normalized_scores))
min_score = np.min(raw_scores)
max_score = np.max(raw_scores)

normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())

print(len(normalized_scores), len(y))
print(normalized_scores[y == 1])

print(sum(y))

print(normalized_scores[y == 0].mean(), normalized_scores[y == 0].std())
print(normalized_scores[y == 1].mean(), normalized_scores[y == 1].std())

"""import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

ocsvm = OneClassSVM(gamma='auto')
ocsvm.fit(x_train)

decision_function = ocsvm.decision_function(x_val)
normalized_scores = 1 / (1 + np.exp(-decision_function))
print(normalized_scores.min(), normalized_scores.max())
normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
print(normalized_scores)"""


sorted_scores = np.sort(normalized_scores)

# Generating cumulative percentages
cumulative_percentages = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

plt.plot(sorted_scores, cumulative_percentages)
plt.title('Cumulative Distribution of Confidence Scores')
plt.xlabel('Risk Score')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.show()