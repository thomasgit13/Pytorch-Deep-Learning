import torch 
from torch import nn 
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
torch.manual_seed(101) 

class CBOW(nn.Module):
    # given a context of some words what is the target
    def __init__(self,embedding_size,vocab_size):
        super(CBOW,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.linear = nn.Linear(embedding_size,vocab_size)
        self.logits = nn.Softmax(dim=1)
    def forward(self,x):
        #shape x : (BATCH_DIM,CONTEXT*2) 
        emb_x = self.embedding(x).mean(1) # (B,EMB_DIM)
        output = self.linear(emb_x) #(BATCH_DIM,VOCAB_DIM)
        probs = self.logits(output)
        return probs 

with open('GitHub/Pytorch-Deep-Learning/dataset/fault_in_our_stars.txt') as file:
    text = file.read()
words = [i.replace('.','') for i in text.replace('/n',' ').split() if len(i)>0]
words = [i.lower() for i in words]
unique_words = list(set(words))
index2word = {ind:word for ind,word in enumerate(unique_words)}
word2index = {j:i for i,j in index2word.items()}

training_pairs = []
for i in range(2,len(words)-2):
    context = words[i-2:i]+words[i+1:i+3]
    context = torch.tensor([word2index[i] for i in context])
    target = words[i]
    target = torch.tensor(word2index[target])
    pair = (context,target)
    training_pairs.append(pair)


class dataset(Dataset):
    def __init__(self,training_pairs):
        self.training_pairs = training_pairs
  
    def __getitem__(self,idx):
        return self.training_pairs[idx][0],self.training_pairs[idx][1]
    def __len__(self):
        return len(self.training_pairs)
    
trainset = dataset(training_pairs)
train_loader = DataLoader(trainset,batch_size=16,shuffle=False)


EMB_DIM = 128
N_EPOCHS = 400
LEARNING_RATE = 0.001
EVAL_INTERVAL = 40 

loss_func = nn.CrossEntropyLoss()
VOCAB_SIZE = len(index2word) 
cbow_model = CBOW(embedding_size=EMB_DIM,vocab_size=VOCAB_SIZE)
optimizer = optim.Adam(cbow_model.parameters(),lr = LEARNING_RATE)
for epoch in range(N_EPOCHS):
    for context,target in train_loader:
        output = cbow_model(context)
        loss = loss_func(output,target)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    if epoch % EVAL_INTERVAL == 0 or iter == epoch - 1:
        print(f'Epoch :{epoch}; loss : {loss} ')


@torch.no_grad()
def evaluate(context_words):
    context_indexes = torch.tensor([word2index[i] for i in context_words])
    context_indexes = context_indexes.unsqueeze(0) 
    probs = cbow_model(context_indexes)
    return probs
    
probs  = evaluate(['communicated',"exclusively","sighs","almost"])
argindex = probs.argsort(descending = True)[0]
final_words = [index2word[i.item()] for i in argindex]
print(final_words)
