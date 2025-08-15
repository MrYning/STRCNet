import pickle
import time
import torch
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import json
import argparse
import random
import sys
import os
from numpy import genfromtxt
from lib.dataloader import normal_and_generate_dataset, get_mask, get_grid_node_map_maxtrix
from lib.early_stop import EarlyStopping
from STRCNet2.STRCNet import STRCNet
from lib.utils import mask_loss, compute_loss, predict_and_evaluate, small_loss

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

city = 'chicago'
grid_small = 10  # it should be 5(k=4) or 10(k=2).
single_model = False
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=f"./config/{city}/GSNet_{city}_Config.json",
                    help='configuration file')
parser.add_argument("--gpus", type=str, default='0', help="test program")
parser.add_argument("--test", action="store_true", help="test program")

# parser.add_argument('--batch', type=int, default=8, metavar='B', help='batch size for training')
# parser.add_argument('--frp', type=int, default=0)
# parser.add_argument('--dataset', type=str, default='')
# parser.add_argument('--SEHeads', type=int, default=1)
# parser.add_argument('--recoupling', type=int, default=1)
# parser.add_argument('--temper', type=float, default=0.5)
# parser.add_argument('--Network', type=str, default='FusionNet')
# parser.add_argument('--knn_attention', type=float, default=0.5)
# parser.add_argument('--epochs', type=int, default=300)

args = parser.parse_args()
print(args.config)
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device=", device)

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']
all_data_filename_small = config['all_data_filename_small']
trans_file = config['trans']

mask_filename = config['mask_filename']
grid_node_filename = config['grid_node_filename']
grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)
num_of_vertices = grid_node_map.shape[1]

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']
data_type = config['data_type']


def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             high_test_loader,
             risk_mask,
             trainer,
             early_stop,
             device,
             scaler,
             data_type=data_type
             ):
    global_step = 1
    for epoch in range(1, training_epoch + 1):
        trans = genfromtxt(trans_file, delimiter=',').astype(np.float32)
        trans = torch.tensor(trans)
        net.train()
        for train_feature, train_feature_small, train_label, train_label_small in train_loader:
            trans, train_feature, train_feature_small, train_label, train_label_small = trans.to(
                device), train_feature.to(device), train_feature_small.to(device), \
                train_label.to(device), train_label_small.to(device)
            # training model
            pred_fg, distill_loss, view_map = net(train_feature, train_feature_small)
            loss = mask_loss(pred_fg, train_label, risk_mask, data_type=data_type) + distill_loss
            # loss = mask_loss(pred_fg, train_label, risk_mask, data_type=data_type)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            global_step += 1

        val_loss = compute_loss(net, val_loader, risk_mask, trans_file, grid_node_map, global_step - 1, device,
                                data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, trans_file, grid_node_map, global_step - 1, scaler,
                                     device)
            high_test_rmse, high_test_recall, high_test_map, _, _ = \
                predict_and_evaluate(net, high_test_loader, risk_mask, trans_file, grid_node_map, global_step - 1,
                                     scaler, device)
            print(
                'global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,hihg test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                % (global_step - 1, epoch, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                   high_test_map), flush=True)

        early_stop(val_loss, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall, high_test_map,
                   test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)
            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                  % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map), flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                  % (early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map), flush=True)
            break

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map


def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    gcn_num_filter = config['gcn_num_filter']

    loaders = []
    scaler = ""
    train_data_shape = ""
    for idx, (x, x_small, y, y_small, high_x, high_x_small, high_y, high_y_small,
              scaler) in enumerate(
        normal_and_generate_dataset(
            all_data_filename,
            all_data_filename_small,
            train_rate=train_rate,
            valid_rate=valid_rate,
            recent_prior=recent_prior,
            week_prior=week_prior,
            one_day_period=one_day_period,
            days_of_week=days_of_week,
            pre_len=pre_len)):

        print("feature:", str(x.shape), "label:", str(y.shape),
              "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape

        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(x_small),
                torch.from_numpy(y),
                torch.from_numpy(y_small)
            ),
            batch_size=batch_size,
            shuffle=(idx == 0),
            drop_last=True
        ))

        # idx为2时是测试数据，以下是高峰期测试数据
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_x_small),
                    torch.from_numpy(high_y),
                    torch.from_numpy(high_y_small)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0),
                drop_last=True
            )
    train_loader, val_loader, test_loader = loaders

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    Model = STRCNet(
        batch_size,
        train_data_shape[2],
        32,
        seq_len,
        8,
        16,
        32,
    )
    Model.to(device)

    num_of_parameters = 0
    for name, parameters in Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)

    risk_mask = get_mask(mask_filename)

    best_mae, best_mse, best_rmse = training(
        Model,
        training_epoch,
        train_loader,
        val_loader,
        test_loader,
        high_test_loader,
        risk_mask,
        trainer,
        early_stop,
        device,
        scaler,
        data_type=data_type
    )
    return best_mae, best_mse, best_rmse


if __name__ == "__main__":
    t_start = time.time()
    main(config)
    t_end = time.time()
    print(f'Running used time: {(t_end - t_start) // 60} min')
