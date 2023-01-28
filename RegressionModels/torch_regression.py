import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from torchData import RegressionDataset
from network import RegressionNetwork

df = pd.read_csv('data.csv')

def get_class_distribution(y):
    element,counts = np.unique(y,return_counts = True)
    output = dict(zip(element,counts))
    return output

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
# data split: X_train,X_test,X_val, y_train,y_test,y_val
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=69
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21
)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# converting data to numpy array 
X_train_scaled, y_train = np.array(X_train_scaled), np.array(y_train)
X_val_scaled, y_val = np.array(X_val_scaled), np.array(y_val)
X_test_scaled, y_test = np.array(X_test_scaled), np.array(y_test)
y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)
 
def get_dataset(x,y):
    x = torch.from_numpy(x).float() 
    y = torch.from_numpy(y).float() 
    dataset = RegressionDataset(x,y) 
    return dataset 

# creating torch data set from numpy array 
train_dataset = get_dataset(X_train_scaled,y_train)
val_dataset = get_dataset(X_val_scaled,y_val)
test_dataset = get_dataset(X_test_scaled,y_test)


EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_FEATURES = len(X.columns)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RegressionNetwork(NUM_FEATURES)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)

# training 
loss_stats = {'train': [],"val": []}
print("Begin training.")
for e in range(1, EPOCHS + 1):

    # TRAINING
    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
    # VALIDATION    
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()
    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))

    print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | '
          f'Val Loss: {val_epoch_loss / len(val_loader):.5f}')


y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


mse = mean_squared_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)
print("Mean Squared Error :",mse)
print("R^2 :",r_square)

