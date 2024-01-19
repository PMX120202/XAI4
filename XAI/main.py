import time
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import torch
from torch import nn, Tensor, optim


import util
# from engine import Engine
from pertubate import FadeMovingAverage1
from graphwavenet.graphwavenet import GraphWaveNet
import argparse
parser = argparse.ArgumentParser()

"""
"""

parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
parser.add_argument('--data', type=str, default='store/METR-LA', help='data path')
parser.add_argument('--adj_data', type=str, default='store/adj_mx.pkl', help='adj data path')

parser.add_argument('--epochs', type=int, default=15, help='')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='saved_models/dynamask', help='save path')
parser.add_argument('--log', type=str, default='log/dynamask', help='save path')

parser.add_argument('--blackbox_file', type=str, default='save_blackbox/G_T_model_1.pth', help='blackbox .pth file')

parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--timestep', type=str, default=12, help='time step')
parser.add_argument('--input_dim', type=str, default=2, help='channels')
parser.add_argument('--output_dim', type=str, default=1, help='channels')
parser.add_argument('--samples', type=str, default=10, help='samples data')
parser.add_argument('--sample_iters', type=str, default=1000, help='___')
parser.add_argument('--initial_mask_coeff', type=str, default=0.5, help='___')
parser.add_argument('--reg_coeff', type=str, default=0.07, help='___')

args = parser.parse_args()




def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adj_data)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size )
    scaler = dataloader['scaler']
    
    permutation = np.random.permutation(len(dataloader['x_test']))
    shuffled_trainx = dataloader['x_test'][permutation]
    samples = shuffled_trainx[:args.samples]
    
    
    edge_index = [[], []]
    edge_weight = []

            
    for i in range(207):
        for j in range(207):
            if adj_mx[(i, j)] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_weight.append(adj_mx[(i, j)])

    edge_index = torch.tensor(edge_index)
    edge_weight = torch.tensor(edge_weight)

    
    # load blackbox

    model = GraphWaveNet(207, 2, 1, 12).to(device)

    model.load_state_dict(torch.load(args.blackbox_file))
   
    adj_mx = torch.tensor(adj_mx).to(device)
    
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    pertubate = FadeMovingAverage1(device)
    
    
    print('start training...', flush=True)
    log_file_test = open('loss_test_log.txt', 'w')
    l2_reg_coeff= 0.08
    
    for i, sample in enumerate(samples):
        print(f'training sample {i}\n' + '=' * 50)
        x = torch.Tensor(sample).to(device)
        print("X: ", x.shape)
        y = model(x, edge_index,edge_weight)
        print("y: ", y.shape)
        
       
        initial_mask = args.initial_mask_coeff * torch.ones(size=x.shape[:-1], device=device)
        mask = initial_mask.clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([mask], lr=args.learning_rate, weight_decay=args.weight_decay)
        
        mae = []
        mape = []
        rmse = []
        
        for iter in range(args.sample_iters):
            optimizer.zero_grad()
            xm = pertubate.apply(x, mask)
            ym = model(xm, edge_index,edge_weight)
            
            loss= util.masked_mae(ym, y)
            l2_reg = l2_reg_coeff * torch.sum(mask**2)
            loss += l2_reg 
            mape_loss = util.masked_mape(ym, y, 0.0)
            rmse_loss= util.masked_rmse(ym, y, 0.0)
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            mae.append(loss.item())
            mape.append(mape_loss.item())
            rmse.append(rmse_loss.item())
            
            
            mask.data = mask.data.clamp(0, 1)
           
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train MAE: {:.4f}, MAPE: ' + '{:.4f}, RMSE: {:.4f}'
                print(log.format(iter, mae[-1], mape[-1], rmse[-1]), flush=True)
            
        mtest_loss = np.mean(mae)
        mtest_mape = np.mean(mape)
        mtest_rmse = np.mean(rmse)
        
        # log = 'sample: {:03d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, ' + 'Test RMSE: {:.4f}'
        print(f'Epoch {i}, Test Loss: {mtest_loss:.4f}, Test MAPE: {mtest_mape:.4f}, Test RMSE: {mtest_rmse:.4f} \n')

        # print(log.format(sample, 
        #                  mtest_loss,
        #                  mtest_mape,
        #                  mtest_rmse)) 
        
        log_file_test.write(f'Epoch {i}, Test Loss: {mtest_loss:.4f}, Test MAPE: {mtest_mape:.4f}, Test RMSE: {mtest_rmse:.4f} \n')
        log_file_test.flush()
            
        y_binary = torch.round(y).squeeze(-1)
        mask = torch.tensor(mask)

 
        mask_cpu = mask.cpu() * np.random.randint(10, 101, size=(12, 207))
        y_binary_cpu = y_binary.cpu()
        print(mask_cpu)
    
        mask_binary = (mask_cpu > 0.5).float()
        y_true = y_binary_cpu.cpu().detach().numpy()
        multi_class_strategy = 'ovr' if len(np.unique(y_true)) == 2 else 'ovo'
        auc_score = roc_auc_score(mask_binary.flatten(), y_true.flatten(), multi_class=multi_class_strategy)
        print(f'AUC Score: {auc_score:.4f}')
        
        
        
        filename = args.save + '/saliency_' + str(i) + '.npz'
        # with open(filename, 'w') as f:
        np.savez(filename, mask.detach().cpu().numpy())


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Total time spent: {:.4f}'.format(t2 - t1))
