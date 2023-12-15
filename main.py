from data_handler import MockDataHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
    "commit_summary",
]

project_name = "Cli"
CREATE_FEATURES = False

# One of if, svm, knn, lin
MODEL = "lin" 

if CREATE_FEATURES:
    y = dh.create_labels(project_name)
    features_per_cid = dh.create_features_from_cc(project_name, requested_features)
    x, y = dh.flatten_features(features_per_cid, y)

    np.save(f"{project_name}_x", x)
    np.save(f"{project_name}_y", y)
else:
    x = np.load(f"{project_name}_x.npy")
    y = np.load(f"{project_name}_y.npy")

    def pick_equal_zeros_ones(x, y):
        count_0 = np.sum(y == 0)
        count_1 = np.sum(y == 1)

        min_count = min(count_0, count_1)

        selected_0s_x = x[y == 0][:min_count]
        selected_0s_y = y[y == 0][:min_count]

        selected_1s_x = x[y == 1][:min_count]
        selected_1s_y = y[y == 1][:min_count]

        concatenated_x = np.concatenate([selected_0s_x, selected_1s_x])
        concatenated_y = np.concatenate([selected_0s_y, selected_1s_y])

        return concatenated_x, concatenated_y


    #x, y = pick_equal_zeros_ones(x, y)

if MODEL == "if":
    iso_forest = IsolationForest(contamination=sum(y)/len(y))
    iso_forest.fit(x)
    raw_scores = iso_forest.decision_function(x)
    print(raw_scores.mean(), raw_scores.std(), raw_scores.min(), raw_scores.max(), np.median(raw_scores))
    normalized_scores = 1 / (1 + np.exp(-raw_scores))
    normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
    normalized_scores = 1 - normalized_scores
elif MODEL == "svm":
    #x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    ocsvm = OneClassSVM(gamma='auto')
    ocsvm.fit(x)

    decision_function = ocsvm.decision_function(x)
    normalized_scores = 1 / (1 + np.exp(-decision_function))
    normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
elif MODEL == "knn":
    K = 10
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(x)

    distances, indices = knn.kneighbors(x)
    anomaly_scores = distances[:, K-1]
    normalized_scores = 1 / (1 + np.exp(-anomaly_scores))
    normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
elif MODEL == "lin":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
    from sklearn.model_selection import KFold

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

    num_epochs = 500
    kfold = KFold(n_splits=5, shuffle=True)

    best_models = []
    normalized_scores = []
    y = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # Split data into training and validation sets
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)

        # Create data loaders for training and validation
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        # Initialize the neural network, loss function, and optimizer for each fold
        model = SimpleLinearNN(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        best_model_state = None

        # Training loop for the current fold
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

            if epoch % 10 == 0:
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

        normalized_scores.extend(predictions)
        y.extend(labels)

    normalized_scores = np.array(normalized_scores)
    y = np.array(y)

print(normalized_scores.mean(), normalized_scores.std(), normalized_scores.min(), normalized_scores.max(), np.median(normalized_scores))
print(normalized_scores[y == 0].mean(), normalized_scores[y == 0].std(), normalized_scores[y == 0].min(), normalized_scores[y == 0].max(), np.median(normalized_scores[y == 0]))
print(normalized_scores[y == 1].mean(), normalized_scores[y == 1].std(), normalized_scores[y == 1].min(), normalized_scores[y == 1].max(), np.median(normalized_scores[y == 1]))


sorted_scores = np.sort(normalized_scores)

# Generating cumulative percentages
cumulative_percentages = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

plt.plot(sorted_scores, cumulative_percentages)
plt.title('Cumulative Distribution of Confidence Scores')
plt.xlabel('Risk Score')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.show()