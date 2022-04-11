import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#%%


class SEblock(nn.Module): #channel attention
    def __init__(self,num_in):
        super(SEblock, self).__init__()  
        self.num_in=num_in
        self.squeeze=nn.AvgPool2d(26) #將特徵圖壓縮成1x1
        self.w1=nn.Sequential(nn.Linear(num_in, num_in//num_in),nn.ReLU(inplace=True)) #取通道間的非線性關係
        self.w2=nn.Sequential(nn.Linear(num_in//num_in, num_in),nn.Softmax(dim=0))    #取通道的softmax(機率)
    def forward(self, x):     
        v=self.squeeze(x) #將輸入壓成通道的形狀 batch , channel , 1 , 1
        v=v.view(-1,self.num_in) # batch , 1
        v=self.w2(self.w1(v)) #attention matrix
        result=x*v.view(-1,self.num_in,1,1)       #原始圖 x attention matrix
        return result
        


#%%
class SpatialAttention(nn.Module): #channel attention
    def __init__(self,num_in):
        super(SpatialAttention, self).__init__()  
        self.num_in=num_in
        self.cnn1 = nn.Conv2d(in_channels=num_in, out_channels=1, kernel_size=7, stride=1, padding=3) #output_shape=(16,24,24)
        self.softmax=nn.Softmax(dim=1)
   
    def forward(self, x):  
        v=torch.mean( x,1) #將通道壓縮成1  batch , 1 , w , h
        v=self.cnn1(x)       
        v=self.softmax(v)
        result=x*v  
        return result
        



#%%
class CNN_Model(nn.Module):
    def __init__(self,outc,kernelsize):
        super(CNN_Model, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=outc, kernel_size=kernelsize, stride=1, padding=0) #output_shape=(16,24,24)
        self.se = SEblock(outc) #對channel取attention計算
        self.sb = SpatialAttention(outc)    #對每個channel的特徵圖取attention計算     
        self.relu1 = nn.ReLU() # activation

    
    def forward(self, x):

        out = self.cnn1(x)

        out = self.relu1(out)

        out=self.se(out)
        out=self.sb(out)        

        out = out.view(out.size(0), -1)

        return out

model = torch.nn.Sequential( CNN_Model(10,3),nn.Linear((10*26*26), 10))    

