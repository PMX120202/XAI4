



# import argparse
# import os
# import time

# import numpy as np
# import torch
# import util
# # from graphwavenet.model import GraphWaveNet
# # from beatsODE.model import BeatsODE
# # from mtgode.model import MTGODE
# from beatsODE2.model import BeatsODE2
# from beatsODE2_1.model import BeatsODE2_1

# from engine import Engine
# # from engine2 import Engine2

# parser = argparse.ArgumentParser()

# """
# """

# parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
# parser.add_argument('--data', type=str, default='store/METR-LA', help='data path')
# parser.add_argument('--adjdata', type=str, default='store/adj_mx.pkl', help='adj data path')

# parser.add_argument('--epochs', type=int, default=1, help='')
# parser.add_argument('--batch_size', type=int, default=16, help='batch size')

# parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
# parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

# parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
# parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--save', type=str, default='saved_models/nbeatsODE', help='save path')
# parser.add_argument('--savelog2_1', type=str, default='Log/BeatsODE2_1', help='save path')
# parser.add_argument('--savelog', type=str, default='Log/BeatsODE2', help='save path')

# parser.add_argument('--context_window', type=int, default=12, help='sequence length')
# parser.add_argument('--target_window', type=int, default=12, help='predict length')
# parser.add_argument('--patch_len', type=int, default=1, help='patch length')
# parser.add_argument('--stride', type=int, default=1, help='stride')
# parser.add_argument('--blackbox_file', type=str, default='save_blackbox/G_T_model_1.pth', help='blackbox .pth file')
# parser.add_argument('--iter_epoch', type=str, default=1, help='using for save pth file')

# parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
# parser.add_argument('--timestep', type=str, default=12, help='time step')
# parser.add_argument('--input_dim', type=str, default=2, help='channels')
# parser.add_argument('--output_dim', type=str, default=2, help='channels')
# parser.add_argument('--hidden', type=str, default=64, help='hidden layers')
# parser.add_argument('--num_layer', type=str, default=4, help='number layers')

# args = parser.parse_args()

 
    
# def main():
#     device = torch.device(args.device)
#     _, _, adj_mx = util.load_adj(args.adjdata)
#     dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    
#     # Mean / std dev scaling is performed to the model output
#     scaler = dataloader['scaler']
    
#     # model = GraphWaveNet(args.num_nodes, args.input_dim, args.output_dim, args.timestep)
#     # model = BeatsODE(device, args.input_dim, args.output_dim, args.timestep)
#     # model = MTGODE(device, args.input_dim, args.timestep, adj_mx, args.timestep)
#     # model = BeatsODE2(in_dim=2, out_dim=2, seq_len=12)
  

#     var =2

#     if var == 1:
#         model = BeatsODE2(in_dim=2, out_dim=1, seq_len=12)
#         log_dir = args.savelog
#     elif var == 2:
#         model = BeatsODE2_1(in_dim=2, out_dim=1, seq_len=12)
#         log_dir = args.savelog2_1


    

   

    
    
#     engine = Engine(scaler,
#                     model,
#                     args.num_nodes, 
#                     args.learning_rate,
#                     args.weight_decay, 
#                     device, 
#                     adj_mx)
#     adj_mx = torch.tensor(adj_mx).to(device)
#     # engine = Engine2(model, scaler, args.learning_rate, args.weight_decay, device)
    
#     if not os.path.exists(args.save):
#         os.makedirs(args.save)
    
#     # if not os.path.exists(log_dir):
#     #     os.makedirs(log_dir)

#     # log_file_train = open(os.path.join(log_dir, 'loss_train_log.txt'), 'w')
#     # log_file_val = open(os.path.join(log_dir, 'loss_val_log.txt'), 'w')
#     # log_file_test = open(os.path.join(log_dir, 'loss_test_log.txt'), 'w')

 

#     # testing
#     engine.model.load_state_dict(torch.load(args.save + "/G_T_model_1.pth"))
#     outputs = []
#     realy = torch.FloatTensor(dataloader['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
#     print("realy ", realy.shape)

#     for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#         testx = torch.FloatTensor(x).transpose(1, 3).to(device)
#         with torch.no_grad():
#             _, preds = engine.model(testx, adj_mx)
#             preds = preds[:, :, :, 0:1]
#             preds= preds.transpose(1, 3)
           
#         outputs.append(preds.squeeze())

#     yhat = torch.cat(outputs, dim=0)
#     yhat = yhat[:realy.size(0), ...]

    
#     print("yhat ", yhat.shape)
#     print("Training finished")

#     amae = []
#     amape = []
#     armse = []
#     for i in range(12):
#         pred = scaler.inverse_transform(yhat[:, :, i])
#         print("pred ", pred.shape)
#         real = realy[:, :, i]
#         print("real ", real.shape)
#         metrics = util.metric(pred, real)
#         log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#         print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
#         amae.append(metrics[0])
#         amape.append(metrics[1])
#         armse.append(metrics[2])

#     log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + \
#         '{:.4f}, Test RMSE: {:.4f}'
#     print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


# if __name__ == "__main__":
#     t1 = time.time()
#     main()
#     t2 = time.time()
#     print("Total time spent: {:.4f}".format(t2 - t1))








import argparse
import os
import time

import numpy as np
import torch
import util
# from graphwavenet.model import GraphWaveNet
# from beatsODE.model import BeatsODE
# from mtgode.model import MTGODE
from NbeatsODE.model import BeatsODE
from engine import Engine
# from engine2 import Engine2

parser = argparse.ArgumentParser()

"""
"""

parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
parser.add_argument('--data', type=str, default='store/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='store/adj_mx.pkl', help='adj data path')

parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='saved_models/nbeatsODE', help='save path')


parser.add_argument('--savelog', type=str, default='Log/BeatsODE', help='save path')

parser.add_argument('--context_window', type=int, default=12, help='sequence length')
parser.add_argument('--target_window', type=int, default=12, help='predict length')
parser.add_argument('--patch_len', type=int, default=1, help='patch length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--blackbox_file', type=str, default='save_blackbox/G_T_model_1.pth', help='blackbox .pth file')
parser.add_argument('--iter_epoch', type=str, default=1, help='using for save pth file')

parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--timestep', type=str, default=12, help='time step')
parser.add_argument('--input_dim', type=str, default=2, help='channels')
parser.add_argument('--output_dim', type=str, default=2, help='channels')
parser.add_argument('--hidden', type=str, default=64, help='hidden layers')
parser.add_argument('--num_layer', type=str, default=4, help='number layers')

args = parser.parse_args()

 
    
def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    
    # Mean / std dev scaling is performed to the model output
    scaler = dataloader['scaler']
    
    # model = GraphWaveNet(args.num_nodes, args.input_dim, args.output_dim, args.timestep)
    # model = BeatsODE(device, args.input_dim, args.output_dim, args.timestep)
    # model = MTGODE(device, args.input_dim, args.timestep, adj_mx, args.timestep)
    # model = BeatsODE2(in_dim=2, out_dim=2, seq_len=12)
  


    model = BeatsODE(in_dim=2, out_dim=1, seq_len=12)
    log_dir = args.savelog
    save_model= args.save



    

   

    
    
    engine = Engine(scaler,
                    model,
                    args.num_nodes, 
                    args.learning_rate,
                    args.weight_decay, 
                    device, 
                    adj_mx)
    
    # engine = Engine2(model, scaler, args.learning_rate, args.weight_decay, device)
    adj_mx = torch.tensor(adj_mx).to(device)
    
    
  
    # testing
    
    engine.model.load_state_dict(torch.load(args.save + "/G_T_model_15.pth"))
   
            
            
    # engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(best_epoch) + ".pth"))
    outputs = []
    outputs1 = []
    realy = torch.FloatTensor(dataloader['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
  

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.FloatTensor(x).transpose(1, 3).to(device)
        testy = torch.FloatTensor(y)
        testy = testy[:, :, :, 0:1].transpose(1, 3)[:, 0, :, :].to(device)
        # yreal= yreal[:realy.size(0), ...]


        with torch.no_grad():
            _, preds = engine.model(testx, adj_mx)
            preds = preds[:, :, :, 0:1]
            # preds= torch.randn(16, 12, 207, 1)
            preds= preds.transpose(1, 3)
           
        outputs.append(preds.squeeze())
        outputs1.append(testy.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    
    yreal = torch.cat(outputs1, dim=0)
   
    yreal= yreal[:realy.size(0), ...]

    
    # print("yhat ", yhat.shape)
    # print("yreal ", yreal.shape)
    print("Training finished")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        # print("pred ", pred.shape)
        real = yreal[:, :, i]
        # print("real ", real.shape)
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + \
        '{:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))