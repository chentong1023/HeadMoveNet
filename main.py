from tqdm import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
from dataset import FixDataset, UnityDataset
from model import HeadPredictModel

def _init_fn(worker_id):
    np.random.seed(123123)
    random.seed(123123)

class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt

def main(args):
    model = HeadPredictModel(30, 40)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightDecay)
    if args.loss == 'L1' or args.loss == 'MAE':
        criterion = nn.L1Loss()
        print('\n==> Loss Function: L1')
    if args.loss == 'MSE' or args.loss == 'L2':
        criterion = nn.MSELoss()
        print('\n==> Loss Function: L2')
    
    # if args.tensorboard:
    #     writer = SummaryWriter('.tensorboard/{}'.format(args.exp_id))
    # else:
    #     writer = None
        
    # train_dataset = FixDataset('head.npy', 'gaze.npy', 40)
    train_dataset = UnityDataset('./predictions/', './uscreen/', True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batchSize, shuffle=True, worker_init_fn=_init_fn
    )
    
    best_loss = 999999
    
    if args.train:
        for epoch in range(args.epochs):
            loss_logger = DataLogger()
            model.train()
            # train_loader = tqdm(train_loader, dynamic_ncols=True)
            
            for i, (inps, his, tar) in enumerate(train_loader):
                inps = inps.cuda()
                his = his.cuda()
                tar = tar.cuda()
                
                output = model(inps, his)
                
                loss = criterion(output, tar)
                loss_logger.update(loss.item(), args.batchSize)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), './best_model.pth')
                
                # train_loader.set_description(
                #         'loss: {loss:.8f}'.format(
                #         loss=loss_logger.avg)
                # )
            if epoch % 100 == 0:
                print(best_loss)
    model.load_state_dict(torch.load(
        './best_model.pth', map_location='cpu'), strict=False)
    model.cuda()
    model.eval()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 1, shuffle=False, worker_init_fn=_init_fn
    )
    
    pred = []
    first = True
    
    for i, (inps, his, tar) in enumerate(train_loader):
        inps = inps.cuda()
        his = his.cuda()
        output = model(inps, his).detach().cpu().numpy()
        print(output)
        
        if first:
            pred = output
            first = False
        else:
            pred = np.concatenate((pred, output))
    np.save('./pred_head.npy', pred)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'FixationNet')
    # the number of total epochs to run
    parser.add_argument('-e', '--epochs', default=1000, type=int,
                        help='number of total epochs to run (default: 30)')
    # the batch size
    parser.add_argument('-b', '--batchSize', default=16, type=int,
                        help='the batch size (default: 512)')
    # the initial learning rate.
    parser.add_argument('--lr', '--learningRate', default=1e-2, type=float,
                        help='initial learning rate (default: 1e-2)')
    parser.add_argument('--weightDecay', '--wd', default=5e-5, type=float,
                        help='weight decay (default: 5e-5)')
    # the loss function.
    parser.add_argument('--loss', default="L1", type=str,
                        help='Different loss to train the network: L1 | L2 (default: L1)')
    # the frequency that we output the loss in an epoch.
    parser.add_argument('--lossFrequency', default=5, type=int,
                        help='the frequency that we output the loss in an epoch (default: 5)')
    parser.add_argument('--train', default=False, action='store_true',
                        help='train the model or not (default: True)')
    main(parser.parse_args())