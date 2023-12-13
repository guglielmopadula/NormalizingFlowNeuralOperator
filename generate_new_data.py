import meshio
import numpy as np
import torch
from tqdm import trange
from torch_geometric.nn.conv import NNConv
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from torch_geometric.nn import pool

points=meshio.read("data/Stanford_Bunny.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)

reference_triangles=meshio.read("data/Stanford_Bunny.stl").cells_dict["triangle"]

x=torch.tensor(points,dtype=torch.float32)
y=x.clone()

BATCH_SIZE=1

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



points_mesh=torch.tensor(points,dtype=torch.float32)
model=AutoEncoder(latent_dim=5)


model=torch.load("model_pyg.pt")



class Generator(nn.Module):
    def __init__(self,latent_dim):
        super(Generator, self).__init__()
        self.model=torch.load("model_pyg.pt")
        self.model.eval()
        self.latent_dim=latent_dim
        self.nf=torch.load("nf.pt")
        self.nf.eval()

    def forward(self, pos):
        x=self.nf.sample(1)[0]
        return self.model.decoder(x,pos)

model=Generator(5)
model.eval()

all_gen=np.zeros((600,len(points),3))
with torch.no_grad():
    for i in range(600):
        all_gen[i]=model(points_mesh).detach().numpy()



np.save("all_points_gen.npy",all_gen)
