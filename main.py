from data_handler import MockDataHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from imblearn.over_sampling import SMOTE


project_name = "Fileupload"
CREATE_FEATURES = False
MODEL = "all" 
global_results = {}

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    #"changed_graph_scores", 
    "global_graph_diffs", 
    #"changed_graph_diffs",
    "commit_summary",
]

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def get_feats(project_name):
    if CREATE_FEATURES or not os.path.exists(f"{project_name}_x.npy"):
        y = dh.create_labels(project_name)
        features_per_cid = dh.create_features_from_cc(project_name)
        x, y = dh.flatten_features(features_per_cid, y)

        np.save(f"{project_name}_x", x)
        np.save(f"{project_name}_y", y)
    else:
        y = dh.create_labels(project_name)
        dh.create_graph_stats(project_name)
        x = np.load(f"{project_name}_x.npy")
        y = np.load(f"{project_name}_y.npy")

    def filter_features(x, requested_features):
        feature_map = {
            "global_graph_scores": slice(0, 4), #12
            "changed_graph_scores": slice(12, 16), #24
            "global_graph_diffs": slice(24, 28), #36
            "changed_graph_diffs": slice(36, 40), #48
            "commit_summary": slice(48, 52)
        }

        columns_to_keep = []
        for feature in requested_features:
            columns_to_keep.extend(range(*feature_map[feature].indices(x.shape[1])))
        filtered_x = x[:, columns_to_keep]

        return filtered_x
        
    filtered_x = filter_features(x, requested_features)
    """for i in filtered_x :
        print(i)"""
    return filtered_x, y

def run_ml_kfold(model):
    val_results = []
    val_labels = []

    for train_index, val_index in kf.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

        if isinstance(model, RandomForestClassifier) or isinstance(model, SVC) or isinstance(model, xgb.XGBClassifier) or isinstance(model, DecisionTreeClassifier):
            model.fit(x_train, y_train)

            if isinstance(model, SVC):
                raw_scores = model.decision_function(x_val)
            else:
                raw_scores = model.predict_proba(x_val)[:, 1]
        else:
            model.fit(x_train)

            if isinstance(model, NearestNeighbors):
                distances, indices = model.kneighbors(x_val)
                raw_scores = np.sum(distances, axis=1)
            elif isinstance(model, LocalOutlierFactor):
                kneighbors = model.kneighbors(x_val, return_distance=False)
                lrd = 1. / np.mean(model._distances_fit_X_[kneighbors, model.n_neighbors - 1], axis=1)
                lrd_ratios_array = lrd / model._lrd[kneighbors].mean(axis=1)

                raw_scores = 1. / lrd_ratios_array
            elif isinstance(model, GaussianMixture):
                raw_scores = model.score_samples(x_val)
            else:
                raw_scores = model.decision_function(x_val)
                raw_scores = -1 * raw_scores
                    

        val_results.extend(raw_scores)
        val_labels.extend(y_val)

    val_results = np.array(val_results)
    val_labels = np.array(val_labels)

    val_results = 1 / (1 + np.exp(-val_results))

    print_results(val_results, val_labels, model)

def run_if():
    model = IsolationForest(contamination="auto", max_features=1)
    print("="*10 + "IF" + "="*10)
    run_ml_kfold(model)

def run_svm():
    model = OneClassSVM(gamma='auto')
    print("="*10 + "SVM" + "="*10)
    run_ml_kfold(model)

def run_knn(K=10):
    model = NearestNeighbors(n_neighbors=K)
    print("="*10 + "KNN" + "="*10)
    run_ml_kfold(model)

def run_lof():
    model = LocalOutlierFactor(n_neighbors=10, contamination='auto')
    print("="*10 + "LOF" + "="*10)
    run_ml_kfold(model)

def run_gmm():
    model = GaussianMixture(n_components=2, random_state=42)
    print("="*10 + "GMM" + "="*10)
    run_ml_kfold(model) 

def run_rf():
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    print("="*10 + "RF" + "="*10)
    run_ml_kfold(model) 

def run_svc():
    model = SVC(kernel='rbf', probability=True)
    print("="*10 + "SVC" + "="*10)
    run_ml_kfold(model) 

def run_xgb():
    model  = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    print("="*10 + "XGB" + "="*10)
    run_ml_kfold(model) 

def run_dt():
    model  = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    print("="*10 + "DT" + "="*10)
    run_ml_kfold(model) 

def print_results(val_results, val_labels, model):
    nonbic_mean = val_results[val_labels == 0].mean()
    bic_mean = val_results[val_labels == 1].mean()
    nonbic_std = val_results[val_labels == 0].std()
    bic_std = val_results[val_labels == 1].std()
    nonbic_median = np.median(val_results[val_labels == 0])
    bic_median = np.median(val_results[val_labels == 1])
    nonbic_var, bic_var = np.var(val_results[val_labels == 0], ddof=1), np.var(val_results[val_labels == 1], ddof=1)

    n_bic = sum(val_labels)
    n_nonbic = len(val_labels) - n_bic
    
    z_score = (nonbic_mean - bic_mean) / ((nonbic_std**2 / n_nonbic) + (bic_std**2 / n_bic))**0.5
    p_value = norm.sf(abs(z_score)) * 2

    pooled_std = np.sqrt(((n_nonbic - 1) * nonbic_var + (n_bic - 1) * bic_var) / (n_nonbic + n_bic - 2))
    cohensd = (nonbic_mean - bic_mean) / pooled_std

    print(f"{nonbic_mean:.3f}, {nonbic_std:.3f}, {nonbic_median:.3f}")
    print(f"{bic_mean.mean():.3f}, {bic_std:.3f}, {bic_median:.3f}")
    print(f"{abs(z_score):.3f}, {p_value:.3f}, {abs(cohensd):.3f}")

    if project_name not in global_results:
        global_results[project_name] = {}
        global_results[project_name][model.__class__] = {
            "nonbic_mean": nonbic_mean,
            "bic_mean": bic_mean,
            "nonbic_std": nonbic_std,
            "bic_std": bic_std,
            "nonbic_median": nonbic_median,
            "bic_median": bic_median,
            "z_score": z_score,
            "p_value": p_value,
            "cohensd": cohensd
        }

x, y = get_feats(project_name)

if MODEL == "if":
    run_if()
elif MODEL == "svm":
    run_svm()
elif MODEL == "knn":
    run_knn()
elif MODEL == "all_ml":
    run_if()
    run_svm()
    run_rf()
    run_svc()
    run_dt()
elif MODEL == "all":
    for pn in ["Cli", "Fileupload", "Beanutils", "Codec"]:
        print("="*30)
        print("= " + pn)
        print("="*30)
        x, y = get_feats(pn)
        run_if()
        run_svm()
        run_rf()
        run_svc()
        run_dt()
elif MODEL == "lin":
    val_results = []
    val_labels = []
    input_dim = x.shape[-1]

    dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    batch_size = 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class SimpleLinearNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleLinearNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()

        def forward(self, x):
            x = self.relu(self.dropout(self.fc1(x)))
            x = self.relu(self.dropout(self.fc2(x)))
            x = self.fc3(x)
            return x.squeeze()

    model = SimpleLinearNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2000

    best_models = []
    normalized_scores = []
    y = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        model = SimpleLinearNN(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

            if epoch % 50 == 0:
                print(f'Fold {fold+1}, Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

        best_models.append(best_model_state)

        # Collect predictions and labels using the best model
        model.load_state_dict(best_model_state)
        model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                predictions.extend(outputs.tolist())
                labels.extend(targets.tolist())

        val_results.extend(predictions)
        val_labels.extend(labels)

    val_results = np.array(val_results)
    val_labels = np.array(val_labels)

    print_results(val_results, val_labels, model)


"""sorted_scores = np.sort(normalized_scores)
cumulative_percentages = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

plt.plot(sorted_scores, cumulative_percentages)
plt.title('Cumulative Distribution of Confidence Scores')
plt.xlabel('Risk Score')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.show()"""