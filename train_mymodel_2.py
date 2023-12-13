from typing import Any
import torch.nn as nn
import meshio
import numpy as np
import torch
from tqdm import trange

points=meshio.read("data/bunny_coarse_train_0.ply").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)

all_points=np.zeros((600,len(points),3))
for i in range(600):
    all_points[i]=meshio.read("data/bunny_coarse_train_"+str(i)+".ply").points

x=torch.tensor(all_points,dtype=torch.float32)
y=x.clone()

BATCH_SIZE=20

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.nn1=nn.Sequential(nn.Linear(6,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100))
        self.nn2=nn.Sequential(nn.Linear(100,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,5))

    def forward(self,x,pos):
        x=torch.cat((x,pos),dim=2)
        x=x.reshape(-1,6)
        x=self.nn1(x)
        x=x.reshape(BATCH_SIZE,-1,100)
        x=torch.mean(x,dim=1)
        x=self.nn2(x)
        return x



class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim=latent_dim
        self.model=nn.Sequential(nn.Linear(3+latent_dim,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.Linear(100,3))


    def forward(self,latent,pos):
        pos=pos.reshape(BATCH_SIZE,-1,3)
        latent=latent.reshape(-1,1,self.latent_dim).repeat(1,pos.shape[1],1)
        x=torch.cat((latent,pos),dim=2)
        x=x.reshape(-1,3+self.latent_dim)
        x=self.model(x)
        x=x.reshape(BATCH_SIZE,-1,3)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder=Encoder(latent_dim)
        self.decoder=Decoder(latent_dim)

    def forward(self,batch):
        x,pos=batch
        latent=self.encoder(x,pos)
        x=self.decoder(latent,pos)
        return x,latent




r=0.015
t=0

points_mesh=torch.tensor(points,dtype=torch.float32)

points_mesh=points_mesh.reshape(1,-1,3).repeat(all_points.shape[0],1,1)

dataset=torch.utils.data.TensorDataset(x,points_mesh)
train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model=AutoEncoder(latent_dim=5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs=100
for epoch in trange(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        x,pos=batch
        x_pred,latent = model(batch)
        loss = torch.linalg.norm(x_pred-x)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        x_pred=x_pred.reshape(x.shape)
        print(torch.linalg.norm(x_pred-x)/torch.linalg.norm(x))

torch.save(model,"model_pyg.pt")


