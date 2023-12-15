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
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
    "commit_summary",
]

project_name = "Fileupload"
CREATE_FEATURES = False

# One of if, svm, knn, lin, all_ml
MODEL = "all_ml" 

if CREATE_FEATURES:
    y = dh.create_labels(project_name)
    features_per_cid = dh.create_features_from_cc(project_name, requested_features)
    x, y = dh.flatten_features(features_per_cid, y)

    np.save(f"{project_name}_x", x)
    np.save(f"{project_name}_y", y)
else:
    x = np.load(f"{project_name}_x.npy")
    y = np.load(f"{project_name}_y.npy")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

def run_ml_kfold(model):
    val_results = []
    val_labels = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]


        if isinstance(model, RandomForestClassifier) or isinstance(model, SVC):
            model.fit(x_train, y_train)

            if isinstance(model, RandomForestClassifier):
                normalized_scores = model.predict_proba(x_val)[:, 1]
            else:
                anomaly_scores = model.decision_function(x_val)
                normalized_scores = 1 / (1 + np.exp(-anomaly_scores))
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
        else:
            model.fit(x_train)

            if isinstance(model, NearestNeighbors):
                distances, indices = model.kneighbors(x_val)
                anomaly_scores = np.sum(distances, axis=1)
                
                normalized_scores = 1 / (1 + np.exp(-anomaly_scores))
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
            elif isinstance(model, LocalOutlierFactor):
                kneighbors = model.kneighbors(x_val, return_distance=False)
                lrd = 1. / np.mean(model._distances_fit_X_[kneighbors, model.n_neighbors - 1], axis=1)
                lrd_ratios_array = lrd / model._lrd[kneighbors].mean(axis=1)
                anomaly_scores = 1. / lrd_ratios_array
                
                normalized_scores = 1 / (1 + np.exp(-anomaly_scores))
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
                normalized_scores = 1 - normalized_scores
            elif isinstance(model, GaussianMixture):
                anomaly_scores = model.score_samples(x_val)
                
                #normalized_scores = 1 / (1 + np.exp(-anomaly_scores))
                normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            else:
                raw_scores = model.decision_function(x_val)
                normalized_scores = 1 / (1 + np.exp(-raw_scores))
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())

                if isinstance(model, IsolationForest):
                    normalized_scores = 1 - normalized_scores

        val_results.extend(normalized_scores)
        val_labels.extend(y_val)

    val_results = np.array(val_results)
    val_labels = np.array(val_labels)

    print_results(val_results, val_labels)

def run_if():
    model = IsolationForest(contamination="auto")
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
    model = lof = LocalOutlierFactor(n_neighbors=10, contamination='auto')
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

def print_results(val_results, val_labels):
    print(f"{val_results[val_labels == 0].mean():.2f}, {val_results[val_labels == 0].std():.2f}, {np.median(val_results[val_labels == 0]):.2f}")
    print(f"{val_results[val_labels == 1].mean():.2f}, {val_results[val_labels == 1].std():.2f}, {np.median(val_results[val_labels == 1]):.2f}")


if MODEL == "if":
    run_if()
elif MODEL == "svm":
    run_svm()
elif MODEL == "knn":
    run_knn()
elif MODEL == "all_ml":
    run_if()
    run_svm()
    #run_lof()
    #run_gmm()
    run_rf()
    run_svc()
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

    print_results(val_results, val_labels)


"""sorted_scores = np.sort(normalized_scores)
cumulative_percentages = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

plt.plot(sorted_scores, cumulative_percentages)
plt.title('Cumulative Distribution of Confidence Scores')
plt.xlabel('Risk Score')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.show()"""