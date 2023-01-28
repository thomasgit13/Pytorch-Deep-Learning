import torch 
from torch import nn 
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed = 100)

# Create *known* parameters
weight_tensor = torch.rand(size = (1,6))
bias = 0.3

# six features and 50 rows
X = torch.rand(size = (50,6)) 
print('Shape of X : ',X.shape)
print('Shape of weight tensor : ',weight_tensor.shape)

y = X.matmul(weight_tensor.T) 
print('Shape of the output after X*W : ',y.shape)
# print(y)

# Create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

torch.manual_seed(seed = 100)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight  = nn.Parameter(torch.rand(size = (1,6)))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self,X:torch.tensor):
        return X.matmul(self.weight.T)+self.bias

model = LinearRegressionModel()
model_initial_parameters = list(model.parameters())
print('Model initial parameters : ',model_initial_parameters)

# Make predictions with model
with torch.inference_mode(): 
    y_preds = model(X_test)


loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.01)

epochs = 100 
train_loss_values = []
test_loss_values = []
epoch_count = [] 

for epoch in range(epochs):
    model.train() 
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test.type(torch.float))

        if epoch % 10 ==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch:{epoch} | MAE Train Loss: {loss} | MAE Test loss: {test_loss}")

print(model.state_dict())
print(weight_tensor)


model.eval()

# Make predictions on the test data
with torch.inference_mode():
    final_predictions = model(X_test)

print(final_predictions)
print(y_test)



# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)


# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.to("cpu")
print(f"Loaded model:\n{loaded_model}")
print(f"Model on device:\n{next(loaded_model.parameters()).device}")


