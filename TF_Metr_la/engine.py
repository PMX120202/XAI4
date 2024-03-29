import torch
import torch.optim as optim

import util

class Engine():
    def __init__(self, scaler, model, num_nodes, lrate, wdecay, device, adj_mx):
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        print('number of parameters:', len(list(self.model.parameters())))
        
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        

        self.edge_index = [[], []]
        self.edge_weight = []

        # The adjacency matrix is converted into an edge_index list
        # in accordance with PyG API
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.adj_mx = torch.tensor(adj_mx).to(device)
        self.edge_index = torch.tensor(self.edge_index).to(device)
        self.edge_weight = torch.tensor(self.edge_weight).to(device)

    def train(self, input, real_val):
        '''
        input: [batch_size, channels, num_nodes, time_steps]
        '''
        self.model.train()
        self.optimizer.zero_grad()
        
        # [batch_size, time_steps, num_nodes, channels]
        # input = input.transpose(-3, -1)
        
        # output = self.model(input, self.edge_index, self.edge_weight)
        backcast, forecast = self.model(input, self.adj_mx)
        forecast = forecast[:, :, :, 0:1]
        backcast = backcast[:, :, :, 0:1]
        # [batch_size, time_steps, num_nodes, channels]
        forecast = forecast.transpose(-3, -1)
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(forecast)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        '''
        input: [batch_size, channels, num_nodes, time_steps]
        '''
        self.model.eval()
        
        # [batch_size, time_steps, num_nodes, channels]
        # input = input.transpose(-3, -1)
        
        # output = self.model(input, self.edge_index, self.edge_weight)
        backcast, forecast = self.model(input, self.adj_mx)
        forecast = forecast[:, :, :, 0:1]
        backcast = backcast[:, :, :, 0:1]
        # [batch_size, time_steps, num_nodes, channels]
        forecast = forecast.transpose(-3, -1)
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(forecast)
        
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
