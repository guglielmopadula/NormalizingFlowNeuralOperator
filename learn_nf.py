import normflows as nf
import torch
base = nf.distributions.base.DiagGaussian(5,trainable=False)
import matplotlib.pyplot as plt
flows = []
num_layers = 32


for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([3, 64, 64, 3], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))


model = nf.NormalizingFlow(base, flows)


# Load data
import numpy as np
latent=np.load("latent.npy")
latent=latent.reshape(latent.shape[0],-1)
latent=torch.tensor(latent).float()


dataset=torch.utils.data.TensorDataset(latent)
loader=torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000
loss_list = []
for epoch in range(num_epochs):
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        loss = model.forward_kld(x[0])
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.3f}'.format(
                epoch+1, num_epochs, i+1, len(loader), loss.item()))
    loss_list.append(loss.item())
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()
torch.save(model, "nf.pt")