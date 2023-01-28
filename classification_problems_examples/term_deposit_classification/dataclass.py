# creating the Dataset class 
# data should be a numpy array 
class TermDepositData:
    def __init__(self,data,feature_labels=None,target_col= 'target'):
        self.data = data 
        self.feature_names =  feature_labels
        self.target_name = target_col

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,indx):
        # returning x,y
        return self.data[indx,:][:-1],self.data[indx,:][-1]
