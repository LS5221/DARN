 # coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
from torch import nn
from utils.data_utils import data_prep, DataSetCatCon, DatasetFLViT, create_dataset_and_evalmetrix, embed_data_mask
from utils.util import Partial_Client_Selection, valid, average_model, compute_weight, average_model_ips
from utils.start_config import initization_configure
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.IPS import sampling, compute_ips

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
criterion2 = nn.MSELoss(reduction='none')
criterion3 = nn.LogSoftmax(dim=1)
def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))


    # Prepare dataset
    create_dataset_and_evalmetrix(args)
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    model_avg = deepcopy(model).cpu()
    X_train_dic, y_train_dic, X_valid_dic, y_valid_dic, X_test_dic, y_test_dic, train_mean_dic, train_std_dic, ips_dic, mask_dic, LR_weight = data_prep(args, args.datamiss, args.datafull)
    # cosine, person = compute_weight(mask_dic, ips_dic)
    # Configuration for FedAVG, prepare model, optimizer, scheduler


    # Train!
    print("=============== Running training ===============")
    loss_fct = torch.nn.CrossEntropyLoss()
    tot_clients = args.dis_cvs_files
    epoch = -1
    if args.y_dim == 2 and args.task == 'binary':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.y_dim > 2 and args.task == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.y_dim  == 'regression':
        criterion = nn.MSELoss().to(args.device)

    if args == 'regression':
        args.dtask = 'reg'
    else:
        args.dtask = 'clf'
    ips = []
    while True:
        epoch += 1
        # randomly select partial clients
        x_test_all = {
            'data': [],
            'mask': [],
            'ips': [],
            'rowips': []
        }
        y_test_all = {'data': []}
        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in args.proxy_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        val_loader_proxy_clients = {}
        x_test_list, y_test_list = [], []
        for cur_single_client in args.proxy_clients:
            args.single_client = cur_single_client
            args.clients_weightes[args.single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens
            X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = X_train_dic[args.single_client], \
                                                                                        y_train_dic[args.single_client], \
                                                                                        X_valid_dic[args.single_client], \
                                                                                        y_valid_dic[args.single_client], \
                                                                                        X_test_dic[args.single_client], \
                                                                                        y_test_dic[args.single_client], \
                                                                                        train_mean_dic[args.single_client], \
                                                                                        train_std_dic[args.single_client]
            x_test_list.append(X_test)
            y_test_list.append(y_test)
            continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
            train_ds = DataSetCatCon(X_train, y_train, args.cat_idxs, args.dtask, continuous_mean_std)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

            # trainset = DatasetFLViT(args, phase='train')
            # train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)
            valid_ds = DataSetCatCon(X_valid, y_valid, args.cat_idxs, args.dtask, continuous_mean_std)
            val_loader_proxy_clients[args.single_client] = DataLoader(valid_ds,
                                                                       batch_size=args.batch_size, shuffle=True,
                                                                       num_workers=0)
            model = model_all[args.single_client]
            model = model.to(args.device).train()
            optimizer = optimizer_all[args.single_client]
            scheduler = scheduler_all[args.single_client]
            if args.decay_type == 'step':
                scheduler.step()

            print('Train the client', cur_single_client, 'of communication round', epoch)

            for inner_epoch in range(args.E_epoch):
                for step, data in enumerate(train_loader):  # batch = tuple(t.to(args.device) for t in batch)
                    optimizer.zero_grad()
                    args.global_step_per_client[args.single_client] += 1
                    x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, row_ips = data[0].to(args.device), \
                                                                                            data[1].to(args.device), \
                                                                                            data[2].to(args.device), \
                                                                                            data[3].to(args.device), \
                                                                                            data[4].to(args.device), \
                                                                                            data[5].to(args.device), \
                                                                                            data[6].to(args.device), \
                                                                                            data[7].to(args.device)
                    # We are converting the data to embeddings in the next step
                    _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                    if args.attentiontype == 'col':
                        reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
                    else:
                        reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, row_ips)
                    y_reps = reps[:, 0, :]
                    cat_outs = model.mlp1(reps[:, :model.num_categories, :])
                    con_outs = model.mlp2(reps[:, model.num_categories:, :])
                    y_outs = model.mlpfory(y_reps)
                    y_outs = model.mlpfory(y_reps)

                    if args.task == 'regression':
                        loss = criterion(y_outs, y_gts)
                    else:
                        loss = criterion(y_outs, y_gts.squeeze())
                    old_params = dict(model.transformer.named_parameters())
                    if len(con_outs) > 0:
                        con_outs = torch.cat(con_outs, dim=1)
                        l2 = criterion2(con_outs, x_cont)
                        l2[con_mask == 0] = 0
                        # l2 = l2 * con_ips
                        l2 = l2.mean()
                        # print(l2)
                    else:
                        l2 = 0
                    l1 = 0
                    # import ipdb; ipdb.set_trace()
                    n_cat = x_categ.shape[-1]
                    # print(cat_outs,len(cat_outs))
                    # print(x_categ,x_categ.shape)
                    reconstruction_errors_cat = torch.zeros(x_categ.shape).to(x_categ.device)
                    for j in range(1, n_cat):
                        log_x = criterion3(cat_outs[j])
                        log_x = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
                        log_x[cat_mask[:, j] == 0] = 0
                        # log_x *= cat_ips[:, j]
                        l1 += abs(sum(log_x) / cat_outs[j].shape[0])
                        # l1 += criterion1(cat_outs[j], x_categ[:, j])
                    # print(loss, l1, l2)
                    # loss += opt.lam2 * l1 + opt.lam3 * l2
                    loss.backward()

                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if not args.decay_type == 'step':
                        scheduler.step()
                    optimizer.step()
                    new_new_params = dict(model.transformer.named_parameters())

                    # for name, value in model_state_dict.items():
                    writer.add_scalar(str(args.single_client) + '/lr', scalar_value=optimizer.param_groups[0]['lr'],
                                      global_step=args.global_step_per_client[args.single_client])
                    writer.add_scalar(str(args.single_client) + '/loss', scalar_value=loss.item(),
                                      global_step=args.global_step_per_client[args.single_client])


                    args.learning_rate_record[args.single_client].append(optimizer.param_groups[0]['lr'])

                    if (step+1) % 10 == 0:
                        print(cur_single_client, step, ':', len(train_loader), 'inner epoch', inner_epoch, 'round', epoch, ':',
                              args.max_communication_rounds, 'loss', loss.item(), 'lr', optimizer.param_groups[0]['lr'])

            # we use frequent transfer of model between GPU and CPU due to limitation of GPU memory
            model.to('cpu')

        ## ---- model average and eval
        flag = True
        for x_test, y_test in zip(x_test_list, y_test_list):
            if flag:
                x_test_all = x_test
                y_test_all = y_test
                flag = False
            else:
                x_test_all['data'] = np.concatenate((x_test_all['data'], x_test['data']), axis=0)
                x_test_all['mask'] = np.concatenate((x_test_all['mask'], x_test['mask']), axis=0)
                x_test_all['ips'] = np.concatenate((x_test_all['ips'], x_test['ips']), axis=0)
                x_test_all['rowips'] = np.concatenate((x_test_all['rowips'], x_test['rowips']), axis=0)
                y_test_all['data'] = np.concatenate((y_test_all['data'], y_test['data']), axis=0)
        # average model
        average_model_ips(args, model_avg, model_all, mask_dic, LR_weight)
        # average_model_ips(args, model_avg, model_all)
        # then evaluate
        test_ds = DataSetCatCon(x_test_all, y_test_all, args.cat_idxs, args.dtask, continuous_mean_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        for cur_single_client in args.proxy_clients:
            args.single_client = cur_single_client
            model = model_all[args.single_client]
            model.to(args.device).eval()
            valid(args, model, val_loader_proxy_clients[args.single_client], test_loader, TestFlag=True)
            model.cpu()

        args.record_val_acc = args.record_val_acc.append(args.current_acc, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)

        tmp_round_acc = [val for val in args.current_test_acc.values() if not val == []]
        writer.add_scalar("test/average_accuracy", scalar_value=np.asarray(tmp_round_acc).mean(), global_step=epoch)

        if args.global_step_per_client[args.single_client] >= args.t_total[args.single_client]:
            break

    writer.close()
    print("================End training! ================ ")

missingrates = [0.5]
mul_datasets = []
reg_datasets = []
bi_datasets = ["News"]
missingtypes = ["mnar_p_"]
ips_nums = [40]
seeds = [0, 149669, 52983, 746806, 639519]
best_valid_accuracy_list = []
best_test_accuracy_list = []
best_test_auroc_list = []

def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="ViT-small",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--FL_platform", type = str, default="ViT-FedAVG", choices=[ "Swin-FedAVG", "ViT-FedAVG", "Swin-FedAVG", "EfficientNet-FedAVG", "ResNet-FedAVG", "saint-FedAVG"],  help="Choose of different FL platform. ")
    parser.add_argument("--dataset", default="News", help="Which dataset.")
    parser.add_argument("--missingrate", default=0.5)
    parser.add_argument("--missingtype", default="mnar_p_")
    parser.add_argument("--ips_num", default=40)
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--cfg",  type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)
    parser.add_argument('--Pretrained', action='store_true', default=False, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")
    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=256, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='1', help="gpu ids: e.g. 0  0,1,2")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization") #99999

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # parser.add_argument("--learning_rate", default=3e-2, type=float, choices=[5e-4, 3e-2, 1e-3],  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    ## FL related parameters
    parser.add_argument("--E_epoch", default=10, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real", "central"], default="split_3", help="Which data partitions to use")

    parser.add_argument('--vision_dset', action='store_true')
    parser.add_argument('--task', default='binary', type=str, choices=['binary', 'multiclass', 'regression'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--c', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='col', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
    parser.add_argument('--dset_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
    parser.add_argument('--active_log', action='store_true')

    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'denoising', 'mask'])  # 选择预训练模式
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--lam4', default=1, type=float)
    parser.add_argument('--lam5', default=1, type=float)
    parser.add_argument('--pretrain_ratio', default=0.5, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

    args = parser.parse_args()
    for dataset in bi_datasets:
        for missingrate in missingrates:
            for missingtype in missingtypes:
                for ips_num in ips_nums:
                    for runs in range(len(seeds)):
                        args.dataset = dataset
                        args.missingrate = missingrate
                        args.missingtype = missingtype
                        args.ips_num = ips_num
                        args.seed = args.set_seed[runs]
                        out = Path(
                            "/data/lsw/data/data/" + args.dataset + "/" + args.missingtype + args.dataset + "_" + str(
                                args.missingrate) + ".csv")
                        data = pd.read_csv(out)
                        X = data.iloc[:, :-1]

                        nunique = X.nunique()
                        types = X.dtypes
                        categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))

                        y = data.iloc[:, -1:].squeeze()
                        if args.task == 'regression':
                            args.y_dim = 1
                        else:
                            args.y_dim = len(np.unique(y.values))
                        for col in X.columns:
                            if types[col] == 'object' or nunique[col] < 100:
                                categorical_indicator[X.columns.get_loc(col)] = True
                        categorical_columns = X.columns[
                            list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                        cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                        cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                        con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                        cat_dims = []
                        for col in categorical_columns:
                            X[col] = X[col].astype("str")

                        temp = X.fillna("MissingValue")
                        nan_mask = temp.ne("MissingValue").astype(int)
                        for col in categorical_columns:
                            #     X[col] = X[col].cat.add_categories("MissingValue")
                            X[col] = X[col].fillna("MissingValue")
                            l_enc = LabelEncoder()
                            X[col] = l_enc.fit_transform(X[col].values)
                            cat_dims.append(len(l_enc.classes_))
                        y = y.values
                        if args.task != 'regression':
                            l_enc = LabelEncoder()
                            y = l_enc.fit_transform(y)
                        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(
                            int)
                        args.cat_dims = cat_dims
                        args.con_idxs = con_idxs
                        args.cat_idxs = cat_idxs
                        model = initization_configure(args)

                        # Training, Validating, and Testing
                        train(args, model)
                        message = '\n \n ==============Start showing final performance ================= \n'
                        message += 'Final union test accuracy is: %2.5f  \n' % \
                                   (np.asarray(list(args.current_test_acc.values())).mean())
                        message += "================ End ================ \n"

                        with open(args.file_name, 'a+') as args_file:
                            args_file.write(message)
                            args_file.write('\n')

if __name__ == "__main__":
    main()
