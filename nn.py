import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from random import randint

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.n_layers = 2
        self.n_linear = 300
        
        self.Conv2D1 = nn.Conv2d(in_channels=1, out_channels=self.n_layers   , kernel_size=3, stride=1, padding=1)
        self.Conv2D2 = nn.Conv2d(self.n_layers, out_channels=self.n_layers**2, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1     = nn.Linear(self.n_layers**2*7*7,self.n_linear)
        self.fc21    = nn.Linear(self.n_linear,5)
        self.fc22    = nn.Linear(self.n_linear,5)
        self.fc3     = nn.Linear(5,self.n_linear)
        self.fc4     = nn.Linear(self.n_linear,self.n_linear*2)
        self.fc5     = nn.Linear(self.n_linear*2,196)
    
    
    def init_data(self,data,cuda):
        if cuda:
            self.inputs_train  = torch.from_numpy(data.x_train).cuda()
            
            self.inputs_test  = torch.from_numpy(data.x_test).cuda()
        else:
            self.inputs_train  = torch.from_numpy(data.x_train)

            self.inputs_test  = torch.from_numpy(data.x_test)
            
    def encoder(self,x):
        x = func.relu(self.Conv2D1(x))
        x = func.relu(self.Conv2D2(x))#.view(-1,self.n_layers**2*14*14)
        x = func.relu(self.maxpool(x)).view(-1,self.n_layers**2*7*7)
        x = func.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)
    
    def reparam(self,mu,logvar):
        sigma = torch.exp(0.5*logvar)
        return mu + torch.randn_like(sigma)*sigma
    
    def decoder(self,z):
        x = func.relu(self.fc3(z))
        x = func.relu(self.fc4(x))
        return torch.sigmoid(self.fc5(x))
    
    def forward(self,x):
        mu, logvar = self.encoder(x)
        z          = self.reparam(mu,logvar)
        return self.decoder(z), mu, logvar
        
    
    def loss_fnc(self, x_rec, x, mu, logvar):  #ELBO
        BCE = func.binary_cross_entropy(x_rec, x.view(-1,196), reduction='sum')          #reconstruction term
        var = logvar.exp()
        KLD = 0.5*torch.sum(1 + logvar - var - mu.pow(2))                                #regularization term
        return (BCE - KLD)/x.size(0)  #
    
    
    def backprop(self, optimizer, n_train):
        self.train()
        optimizer.zero_grad()
        
        args_batch = randint(0, len(self.inputs_train)-n_train)
        x_recs, mus, logvars = self(self.inputs_train[args_batch: args_batch+n_train])
        
        loss       = self.loss_fnc(x_recs, self.inputs_train[args_batch: args_batch+n_train], mus, logvars)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    
    def test(self, n_test):
        #self.eval()
        with torch.no_grad():
            args_batch = randint(0, len(self.inputs_test)-n_test)
            x_recs, mus, logvars = self(self.inputs_test[args_batch: args_batch+n_test])
            loss       = self.loss_fnc(x_recs, self.inputs_test[args_batch: args_batch+n_test], mus, logvars)
        return loss.item()
    
    
    def predict_test(self,n,path, cuda):
        #self.eval()
        with torch.no_grad():
            
            z_test      = torch.randn(n,5)
            if cuda == 1:
                x_rec       = self.decoder(z_test.cuda())
            else:
                x_rec       = self.decoder(z_test)
                
            x_rec_image = x_rec.view(-1,14,14).cpu().numpy()
            
            for i in range(0,n):
                fig,ax = plt.subplots()
                ax.imshow(x_rec_image[i,:,:], cmap=cm.binary)
                fig.savefig(path+'{}.pdf'.format(i+1))
                plt.close('all')
            