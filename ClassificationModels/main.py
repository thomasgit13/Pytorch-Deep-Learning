import torch
import sklearn.model_selection
import pandas as pd 
import numpy as np 
from dataclass import ClassifierDataset
from network import MulticlassClassification
from func_utils import multiclass_accuracy
import sklearn.metrics as metrics 


data = pd.read_csv('data.csv')
data.quality.value_counts()

target_labels = data.quality.unique()
target_labels = np.sort(target_labels)
target_new_labels = np.argsort(target_labels)
target_map = {i:j for i,j in zip(target_labels,target_new_labels)}

# target is modified 
data.quality = data.quality.replace(target_map)
target_value_counts = data.quality.value_counts()

# separating features and target
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(
    X,y,stratify = y, random_state = 1,test_size = 0.3
)
X_test,X_val,y_test,y_val = sklearn.model_selection.train_test_split(
    X_test,y_test,stratify = y_test,random_state = 1, test_size = 0.1
)

scaler = sklearn.preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

y_train, y_test, y_val =  np.array(y_train),np.array(y_test),np.array(y_val)


train_dataset = ClassifierDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = ClassifierDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = ClassifierDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

# distribution of unique values in a numpy array 
element,count = np.unique(y_train,return_counts=True)
class_counts_object = dict(zip(element,count))
class_weights_object = dict(zip(element,1.0/count))
class_weights = [i for i in class_weights_object.values()]
class_weights = torch.FloatTensor(class_weights)
class_weights_all = [class_weights_object[i] for i in y_train]

weighted_sampler = torch.utils.data.WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

epochs = 300
batch_size = 16
learning_rate = 0.0007
num_features = len(X.columns)
num_classes = len(class_counts_object.keys())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=weighted_sampler
)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MulticlassClassification(num_feature = num_features, num_class=num_classes)
model.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


accuracy_stats = {'train': [],"val": []}
loss_stats = {'train': [],"val": []}
epochs = 128
print("######################  Begin training ################## ")
for e in (range(1, epochs+1)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader: # train loader 
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)  # to device 
        optimizer.zero_grad() # making gradients zero at the start of the each training batch  
        
        y_train_pred = model(X_train_batch) # model output 
        
        train_loss = criterion(y_train_pred, y_train_batch) # finding the loss 
        train_acc = multiclass_accuracy(y_train_pred, y_train_batch)  # accuracy 
        
        train_loss.backward()   # computing the partial derivates 
        optimizer.step() # updating the weights 
        
        train_epoch_loss += train_loss.item()  
        train_epoch_acc += train_acc.item()
    # completed the model training of first batch    
        
    # VALIDATION    
    with torch.no_grad(): # gradients are not needed 
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        # for testing and validation data the batch size is 1 
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multiclass_accuracy(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            
    # appending all the loss and accuracy values for the current batch 
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    
                              
    print(f'Epoch:{e+0:03}\
          | Train Loss: {train_epoch_loss/len(train_loader):.3f}\
              | Val Loss: {val_epoch_loss/len(val_loader):.3f}\
          | Train Acc: {train_epoch_acc/len(train_loader):.3f}\
              | Val Acc: {val_epoch_acc/len(val_loader):.3f}'
        )
print('############### Training Completed #####################')

# Create dataframes
from matplotlib import pyplot as plt
import seaborn as sns
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()

# testing loop 
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_preds = model(X_batch)
        max_values, y_pred_tags = torch.max(y_test_preds, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

class2idx = {3:0,4:1,5:2,6:3,7:4,8:5}
idx2class = {v: k for k, v in class2idx.items()}
data['quality'].replace(class2idx, inplace=True)
confusion_matrix_df = pd.DataFrame(
    metrics.confusion_matrix(y_test, y_pred_list)).rename(
        columns=idx2class, 
        index=idx2class
    )
sns.heatmap(confusion_matrix_df, annot=True,cmap = 'twilight_shifted_r')
plt.show() 

print(metrics.classification_report(y_test, y_pred_list))
