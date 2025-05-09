from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error


import torch
from torch import nn
import torch.nn.functional as F
from utils.scheduler import setup_scheduler
from utils.data_utils import embed_data_mask
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
## for optimizaer

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(str(args.single_client)).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)



def inner_valid(args, model, test_loader):
    eval_losses = AverageMeter()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(args.device)
    y_pred = torch.empty(0).to(args.device)
    prob = torch.empty(0).to(args.device)
    print("++++++ Running Validation of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, data in enumerate(test_loader):

        x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, row_ips = data[0].to(args.device), \
                                                                                data[1].to(args.device), \
                                                                                data[2].to(args.device), \
                                                                                data[3].to(args.device), \
                                                                                data[4].to(args.device), \
                                                                                data[5].to(args.device), \
                                                                                data[6].to(args.device), \
                                                                                data[7].to(args.device)

        with torch.no_grad():
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
            if args.attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, row_ips)
            y_reps = reps[:, 0, :]

            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)
            if args.task == 'binary' or args.task == 'multiclass':
                prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100
    auc = 0
    if args.task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    elif args.task == 'multiclass':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu(), multi_class='ovo')
    return acc.cpu().numpy(), auc

def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag


def classification_scores(model, dloader, device, task, attentiontype):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(
                device), data[2].to(device), data[3].to(device), data[4].to(device)
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
            if attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc)
            # reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)
            if task == 'binary' or task == 'multiclass':
                prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    elif task == 'multiclass':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu(), multi_class='ovo')
    return acc.cpu().numpy(), auc

def mean_sq_error(model, dloader, device):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace()
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        R2 = r2_score(y_test.cpu(), y_pred.cpu())
        rmse = float(rmse)
        R2 = float(R2)
        return rmse, R2

def valid(args, model, val_loader,  test_loader = None, TestFlag = False):
    # Validation!
    if args.task == "regression":
        valid_rmse, valid_R2 = mean_sq_error(model, val_loader, args.device)
        test_rmse, test_R2 = mean_sq_error(model, test_loader, args.device)
    else:
        accuracy, auroc = classification_scores(model, val_loader, args.device, args.task, args.attentiontype)
        # print('val_acc:', accuracy, 'val_auc:', auroc)
        test_accuracy, test_auroc = classification_scores(model, test_loader, args.device, args.task,
                                                          args.attentiontype)
        print('test_acc:', test_accuracy, 'test_auc:', test_auroc)
        if args.best_acc[args.single_client] < accuracy:
            # if args.best_acc[args.single_client] < eval_result:
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = auroc
            args.best_eval_loss[args.single_client] = accuracy
            print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])

            if TestFlag:
                test_accuracy, test_auroc = classification_scores(model, test_loader, args.device, args.task, args.attentiontype)
                print('test_acc:', test_accuracy, 'test_auc:', test_auroc)
                args.current_test_acc[args.single_client] = test_accuracy
                args.current_test_auc[args.single_client] = test_auroc
                print('We also update the test acc of client', args.single_client, 'as',
                      args.current_test_acc[args.single_client])
        else:
            print("Donot replace previous best metric of client", args.best_acc[args.single_client])

    # eval_result, eval_losses = inner_valid(args, model, val_loader)
    #
    # print("Valid Loss: %2.5f" % eval_losses.avg, "Valid metric: %2.5f" % eval_result)
    # if args.dataset == 'CelebA':
    #     if args.best_eval_loss[args.single_client] > eval_losses.val:
    #         # if args.best_acc[args.single_client] < eval_result:
    #         if args.save_model_flag:
    #             save_model(args, model)
    #
    #         args.best_acc[args.single_client] = eval_result
    #         args.best_eval_loss[args.single_client] = eval_losses.val
    #         print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])
    #
    #         if TestFlag:
    #             test_result, eval_losses = inner_valid(args, model, test_loader)
    #             args.current_test_acc[args.single_client] = test_result
    #             print('We also update the test acc of client', args.single_client, 'as',
    #                   args.current_test_acc[args.single_client])
    #     else:
    #         print("Donot replace previous best metric of client", args.best_acc[args.single_client])
    # else:  # we use different metrics
    #     # if args.best_acc[args.single_client] < eval_result:
    #     if metric_evaluation(args, eval_result):
    #         if args.save_model_flag:
    #             save_model(args, model)
    #
    #         args.best_acc[args.single_client] = eval_result
    #         args.best_eval_loss[args.single_client] = eval_losses.val
    #         print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])
    #
    #         if TestFlag:
    #             test_result, eval_losses = inner_valid(args, model, test_loader)
    #             args.current_test_acc[args.single_client] = test_result
    #             print('We also update the test acc of client', args.single_client, 'as',
    #                   args.current_test_acc[args.single_client])
    #     else:
    #         print("Donot replace previous best metric of client", args.best_acc[args.single_client])
    #
    # args.current_acc[args.single_client] = eval_result


def optimization_fun(args, model):

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")
    return optimizer


def Partial_Client_Selection(args, model):

    # Select partial clients join in FL train
    if args.num_local_clients == -1: # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients = len(args.dis_cvs_files)# update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    scheduler_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    for proxy_single_client in args.proxy_clients:
        model_all[proxy_single_client] = deepcopy(model).cpu()
        optimizer_all[proxy_single_client] = optimization_fun(args, model_all[proxy_single_client])

        # get the total decay steps first
        if not args.dataset == 'CelebA':
            args.t_total[proxy_single_client] = args.clients_with_len[proxy_single_client] *  args.max_communication_rounds / args.batch_size * args.E_epoch
        else:
            # just approximate to make sure average communication round for each client is args.max_communication_rounds
            tmp_rounds = [math.ceil(len/32) for len in args.clients_with_len.values()]
            args.t_total[proxy_single_client]= sum(tmp_rounds)/(args.num_local_clients-1) *  args.max_communication_rounds
        scheduler_all[proxy_single_client] = setup_scheduler(args, optimizer_all[proxy_single_client], t_total=args.t_total[proxy_single_client])
        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    return model_all, optimizer_all, scheduler_all
def compute_cosine_similarity(A, B):
    return 1 - cosine(A.flatten(), B.flatten())

def compute_pearson_correlation(A, B):
    return pearsonr(A.flatten(), B.flatten())[0]

def compute_weight(mask_dic, weight_dic):
    mask_w = {}
    for key, df in mask_dic.items():
        # 对每个 DataFrame 按列求和
        summed_df = df.values.sum()
        mask_w[key] = summed_df
    num_tensors = len(weight_dic)
    cosine_similarity_matrix = torch.zeros((num_tensors, num_tensors))
    pearson_correlation_matrix = torch.zeros((num_tensors, num_tensors))

    tensor_values = list(weight_dic.values())

    # 计算相似度
    for i in range(num_tensors):
        for j in range(i, num_tensors):
            if i != j:
                cosine_similarity_matrix[i, j] = 0.5 * (1 - compute_cosine_similarity(tensor_values[i], tensor_values[j]))
                pearson_correlation_matrix[i, j] = 0.5 * (1 - compute_pearson_correlation(tensor_values[i], tensor_values[j]))


    return mask_w, cosine_similarity_matrix, pearson_correlation_matrix


def average_model_ips(args, model_avg, model_all, mask_dic, LR_weight):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())
    mask_w, cosine_similarity_matrix, pearson_correlation_matrix = compute_weight(mask_dic, LR_weight)
    cosine_similarity_matrix = (cosine_similarity_matrix + cosine_similarity_matrix.t())
    total_sum = sum(mask_w.values())
    weight_cos = torch.sum(cosine_similarity_matrix, dim=1)
    # 使用每个值除以总和
    mask_w = {key: value / total_sum for key, value in mask_w.items()}

    # for name, value in model_state_dict.items():
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()

            if client == 0:
                if weight_cos[client] < 0.018 * (len(weight_cos) - 1):
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
            else:
                if weight_cos[client] < 0.018 * (len(weight_cos) - 1):
                    try:
                        tmp_param_data = tmp_param_data + \
                                         dict(model_all[single_client].named_parameters())[
                                             name].data * single_client_weight
                    except:
                        tmp_param_data = dict(model_all[single_client].named_parameters())[
                                             name].data * single_client_weight
        params[name].data.copy_(tmp_param_data)

    print('Update each client model parameters----')
    for single_client in args.proxy_clients:
        tmp_params = dict(model_all[single_client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)

    # alpha = 0.5
    # beta = 0.5
    # for now_ave_client in args.proxy_clients:
    #     for name, param in params.items():
    #         for now_client in args.proxy_clients:
    #
    #             if now_client == 0:
    #                 if now_client == now_ave_client:
    #                     tmp_param_data = dict(model_all[now_ave_client].named_parameters())[
    #                                          name].data * alpha
    #                 else:
    #                     tmp_param_data = (dict(model_all[now_client].named_parameters())[
    #                                          name].data * (mask_w[now_client] * beta + (1 - beta) * cosine_similarity_matrix[now_ave_client][now_client])) * (1 - alpha)
    #             else:
    #                 if now_client == now_ave_client:
    #                     tmp_param_data += dict(model_all[now_ave_client].named_parameters())[
    #                                          name].data * alpha
    #                 else:
    #                     tmp_param_data += (dict(model_all[now_client].named_parameters())[
    #
    #                                          name].data * (mask_w[now_client] * beta + (1 - beta) * cosine_similarity_matrix[now_ave_client][now_client])) * (1 - alpha)
    #     tmp_params = dict(model_all[now_ave_client].named_parameters())
    #
    #     tmp_params[name].data.copy_(tmp_param_data)
    # for single_client in args.proxy_clients:
    #     for name, param in params.items():
    #         for client in range(len(args.proxy_clients)):
    #             if client == 0:
    #                 if client == single_client:
    #                     tmp_param_data = dict(model_all[single_client].named_parameters())[
    #                                          name].data * alpha
    #                 else:
    #                     tmp_param_data = (dict(model_all[single_client].named_parameters())[
    #                                          name].data) * (1 - alpha)
    #             else:
    #
    #             now_single_client = args.proxy_clients[client]
    #
    #             single_client_weight = args.clients_weightes[now_single_client]
    #             single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
    #
    #             if client == 0:
    #                 tmp_param_data = dict(model_all[single_client].named_parameters())[
    #                                      name].data * single_client_weight
    #             else:
    #                 tmp_param_data = tmp_param_data + \
    #                                  dict(model_all[single_client].named_parameters())[
    #                                      name].data * single_client_weight
    #         params[name].data.copy_(tmp_param_data)

def average_model(args, model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())

    # for name, value in model_state_dict.items():
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()

            if client == 0:
                tmp_param_data = dict(model_all[single_client].named_parameters())[
                                     name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(model_all[single_client].named_parameters())[
                                     name].data * single_client_weight
        params[name].data.copy_(tmp_param_data)

    print('Update each client model parameters----')

    for single_client in args.proxy_clients:
        tmp_params = dict(model_all[single_client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)
