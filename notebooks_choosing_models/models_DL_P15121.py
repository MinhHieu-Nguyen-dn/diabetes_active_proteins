import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ray import tune
from ray import train as raytrain
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float().to(device)
        self.target = torch.from_numpy(target).float().to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


def string_to_np_array(s: str):
    s = s.strip('[]')
    arr = np.fromstring(s, sep=' ')

    return arr


df = pd.read_csv('data/processed_P15121.csv', index_col=0)
df['fp'] = df['fp'].apply(string_to_np_array)
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['fp'].tolist(), df['active'].tolist(), test_size=0.2,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Creating the datasets
train_dataset = MyDataset(np.array(X_train), np.array(y_train))
val_dataset = MyDataset(np.array(X_val), np.array(y_val))
test_dataset = MyDataset(np.array(X_test), np.array(y_test))

# Creating the dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define the FCNN
class Net(nn.Module):
    def __init__(self, num_hidden_layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        input_units = 167
        for i in range(num_hidden_layers):
            output_units = max(input_units // 2, 1)
            self.layers.append(nn.Linear(input_units, output_units))
            input_units = output_units
        self.fc_out = nn.Linear(input_units, 1)
        self.to(device)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return torch.sigmoid(self.fc_out(x))


# Define a training step
def train(model, optimizer, criterion, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    target = target.view(-1, 1).float()  # Reshape the target to match the input shape
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


# Define a validation step
def validate(model, criterion, data, target):
    model.eval()
    with torch.no_grad():
        output = model(data)
        target = target.view(-1, 1)
        loss = criterion(output, target)
        return loss.item()


# Define a function for hyperparameter tuning
def tune_model(config):
    model = Net(num_hidden_layers=config["hidden_layers"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(10):
        loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            data, target = batch
            loss += train(model, optimizer, criterion, data, target)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        loss /= len(train_loader)
        accuracy = correct / total
        train_losses.append(loss)
        train_accuracies.append(accuracy)

        val_loss = 0
        correct = 0
        total = 0
        for batch in val_loader:
            data, target = batch
            val_loss += validate(model, criterion, data, target)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        val_loss /= len(val_loader)
        accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

    train_losses = sum(train_losses)/len(train_losses)
    val_losses = sum(val_losses)/len(val_losses)
    train_accuracies = sum(train_accuracies)/len(train_accuracies)
    val_accuracies = sum(val_accuracies)/len(val_accuracies)

    raytrain.report({
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies
    })


# Perform hyperparameter tuning
analysis = tune.run(
    tune_model,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "hidden_layers": tune.choice([1, 2, 3, 4, 5])
    },
    resources_per_trial={"cpu": 1, "gpu": 1},
    num_samples=10,
    metric="val_loss",
    mode="min"
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="val_loss", mode="min")

# Train the final model with the best hyperparameters
model = Net(num_hidden_layers=best_config["hidden_layers"])
optimizer = optim.Adam(model.parameters(), lr=best_config["lr"])
criterion = nn.BCELoss()
train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)

for epoch in range(50):
    for batch in train_loader:
        data, target = batch
        train(model, optimizer, criterion, data, target)

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        y_true.extend(target.tolist())
        y_pred.extend(output.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred.round())
confusion = confusion_matrix(y_true, y_pred.round())
tn, fp, fn, tp = confusion.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
auc = roc_auc_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')
print(f'AUC: {auc}')

# Save the model
torch.save(model.state_dict(), "models/FCNN_tuned.pt")
