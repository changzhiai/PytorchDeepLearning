# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:49:08 2023

@author: changai
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def get_data():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    
    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    return x_train, y_train

def train():
    input_dim = 1
    output_dim = 1
    
    model = LinearRegressionModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    x_train, y_train = get_data()
    epochs = 100
    for epoch in range(epochs):
        epoch += 1
        # Convert numpy array to torch Variable
        inputs = torch.from_numpy(x_train).requires_grad_()
        labels = torch.from_numpy(y_train)
    
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad() 
    
        # Forward to get output
        outputs = model(inputs)
    
        # Calculate Loss
        loss = criterion(outputs, labels)
    
        # Getting gradients w.r.t. parameters
        loss.backward()
    
        # Updating parameters
        optimizer.step()
    
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        
        save_model = True
        if save_model:
            # Saves only parameters
            # alpha & beta
            torch.save(model.state_dict(), 'awesome_model.pkl')

def predict():
    load_model = True
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    if load_model:
        model.load_state_dict(torch.load('awesome_model.pkl'))
    x_train, y_train = get_data()
    predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(predicted)
    
    plot = True
    if plot:
        # Clear figure
        plt.clf()
        # Get predictions
        fig = plt.figure(dpi=300)
        predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
        
        # Plot true data
        plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
        
        # Plot predictions
        plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
        
        # Legend and plot
        plt.legend(loc='best')
        plt.show()
        fig.savefig('linear_regression.png')
    return predicted

    
if __name__ == '__main__':
    train()
    predict()