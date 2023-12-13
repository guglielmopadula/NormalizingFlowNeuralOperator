from typing import Any
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import pool
import torch.nn as nn
import meshio
import numpy as np
import torch
from tqdm import trange
from torch_geometric.nn.conv import NNConv  

points=meshio.read("data/Stanford_Bunny_red.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)

all_points=np.zeros((600,len(points),3))
for i in range(600):
    all_points[i]=meshio.read("data/bunny_coarse_train_"+str(i)+".ply").points

x=torch.tensor(all_points,dtype=torch.float32)
y=x.clone()

BATCH_SIZE=1

class GNOEnc(nn.Module):
    def __init__(self,latent_dim):
        super(GNOEnc, self).__init__()
        self.in_channels=3
        self.out_channels=latent_dim
        self.latent_dim=latent_dim
        self.nn=nn.Sequential(nn.Linear(12,20),nn.ReLU(),nn.Linear(20,20),nn.ReLU(),nn.Linear(20,20),nn.ReLU(),nn.Linear(20,self.in_channels*self.out_channels))
        self.nn_conv=NNConv(self.in_channels,self.out_channels,self.nn,aggr="mean")
        self.final_nn=nn.Sequential(nn.Linear(latent_dim,50),nn.BatchNorm1d(50),nn.ReLU(),nn.Linear(50,50),nn.BatchNorm1d(50),nn.ReLU(),nn.Linear(50,50))
    
    def forward(self,batch):
        x=self.nn_conv(batch.x,batch.edge_index,batch.edge_attr)
        x=x.reshape(BATCH_SIZE,-1,self.latent_dim)
        len_points=x.shape[1]
        len_graph=batch.edge_index.shape[1]//BATCH_SIZE
        x=torch.mean(x,dim=1)
        x=self.final_nn(x)
        x_2=x.reshape(BATCH_SIZE,-1,self.latent_dim).repeat(1,len_points,1).reshape(-1,self.latent_dim)
        x_3=x.reshape(BATCH_SIZE,-1,self.latent_dim).repeat(1,len_graph,1).reshape(-1,self.latent_dim)
        edge_attr=torch.concatenate((batch.edge_attr[:,:6],x_3),dim=1)
        return x_2,edge_attr,x


class GNODec(nn.Module):
    def __init__(self,latent_dim):
        super(GNODec, self).__init__()
        self.in_channels=latent_dim
        self.out_channels=3
        self.nn=nn.Sequential(nn.Linear(6+latent_dim,20),nn.ReLU(),nn.Linear(20,20),nn.ReLU(),nn.Linear(20,20),nn.ReLU(),nn.Linear(20,self.in_channels*self.out_channels))
        self.nn_conv=NNConv(self.in_channels,self.out_channels,self.nn,aggr="mean")
    
    def forward(self,x,edge_index,edge_attr):
        x=self.nn_conv(x,edge_index,edge_attr)
        return x




class Model(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.gno_enc=GNOEnc(latent_dim)
        self.gno_dec=GNODec(latent_dim)

    def forward(self,batch):
        x,edge_attr,latent=self.gno_enc(batch)
        batch=self.gno_dec(x,batch.edge_index,edge_attr)
        return batch,latent

r=0.015
t=0

points_mesh=torch.tensor(points,dtype=torch.float32)

mylist=pool.radius(points_mesh,points_mesh,r)
train_dataset=[]
for i in range(600):
    tmp=x[i].reshape(-1,3)
    train_dataset.append(Data(x=tmp, y=y[i].reshape(-1,3), edge_index=mylist, edge_attr=torch.cat((points_mesh[mylist[0]],points_mesh[mylist[1]],tmp[mylist[0]],tmp[mylist[1]]),dim=1)))

train_dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

model=Model(latent_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs=10
for epoch in trange(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        x_pred = model(batch)
        loss = torch.linalg.norm(x_pred-batch.x)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print(torch.linalg.norm(x_pred-batch.x)/torch.linalg.norm(batch.x))

torch.save(model,"model_pyg.pt")
all_points_rec=np.zeros((600,len(points),3))
latent_all=np.zeros((600,5))
model=torch.load("model_pyg.pt")
model.eval()
j=0
for batch in train_dataloader:
    x_pred,latent = model(batch)
    all_points_rec[j]=x_pred.reshape(1,-1,3).detach().numpy()
    latent_all[j]=latent.detach().numpy()
    j=j+1

print(np.mean(np.var(all_points_rec,axis=0)))
print(np.mean(np.var(all_points,axis=0)))
np.save("all_points_coarse_train_rec.npy",all_points_rec)
np.save("latent.npy",latent_all)

