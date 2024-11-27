import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import torch
from torchvision import transforms
from torch import nn
import torch.utils.data as data
import random
from .IPS import sampling, compute_ips
Image.LOAD_TRUNCATED_IMAGES = True

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)
col_softmax = nn.Softmax(dim=1)
row_softmax = nn.Softmax(dim=0)

def data_prep(args, data, datafull, datasplit=[.8, .1, .1]):
    X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, ips_dict, mask_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    np.random.seed(args.seed)
    X = data.iloc[:, :-1]
    nunique = X.nunique()
    types = X.dtypes
    categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
    for col in X.columns:
        if types[col] == 'object' or nunique[col] < 100:
            categorical_indicator[X.columns.get_loc(col)] = True

    args.categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    args.cont_columns = list(set(X.columns.tolist()) - set(args.categorical_columns))

    args.cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    args.con_idxs = list(set(range(len(X.columns))) - set(args.cat_idxs))

    y = data.iloc[:, -1:].squeeze()
    for single_client in args.proxy_clients:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1:].squeeze()
        args.single_client = single_client
        X = X.iloc[args.data_all[args.single_client], :]
        y = y.iloc[args.data_all[args.single_client]]
        X.index = range(0, X.shape[0])
        y = y.values
        if args.task != 'regression':
            l_enc = LabelEncoder()
            y = l_enc.fit_transform(y)
        # 1为obsevable，0为miss
        temp = X.fillna("MissingValue")
        nan_mask = temp.ne("MissingValue").astype(int)
        mask_dict[args.single_client] = nan_mask
        # nan_mask 中0为空，1为非空
        for col in args.categorical_columns:
            X[col] = X[col].astype("str")
        cat_dims = []
        # 填补缺失值
        for col in args.categorical_columns:
            #     X[col] = X[col].cat.add_categories("MissingValue")
            X[col] = X[col].fillna("MissingValue")
            l_enc = LabelEncoder()
            X[col] = l_enc.fit_transform(X[col].values)
            cat_dims.append(len(l_enc.classes_))
        X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

        train_indices = X[X.Set == "train"].index
        valid_indices = X[X.Set == "valid"].index
        test_indices = X[X.Set == "test"].index



        X = X.drop(columns=['Set'])
        datamiss = X
        datamiss, indicator = sampling(datafull.iloc[args.data_all[args.single_client], :-1], datamiss, args.ips_num, method='feature')
        # TODO：现在需要每个client的ips矩阵来计算各个client之间的互补性，基于互补性计算权重
        # ips计算，ips计算这里
        ips = compute_ips(datamiss, indicator, num=args.ips_num, method='xgb')
        ips = torch.tensor(ips)
        row_ips = torch.sum(ips, dim=1)
        rowips = row_softmax(row_ips)
        ips = col_softmax(ips)
        ips_dict[args.single_client] = ips
        ips = pd.DataFrame(ips)
        rowips = pd.DataFrame(rowips)
        for col in args.cont_columns:
        #     X[col].fillna("MissingValue",inplace=True)
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
        X_train[args.single_client], y_train[args.single_client] = data_split(X, y, nan_mask, ips, rowips, train_indices)
        X_valid[args.single_client], y_valid[args.single_client] = data_split(X, y, nan_mask, ips, rowips, valid_indices)
        X_test[args.single_client], y_test[args.single_client] = data_split(X, y, nan_mask, ips, rowips, test_indices)

        train_mean[args.single_client], train_std_s = np.array(X_train[args.single_client]['data'][:,args.con_idxs],dtype=np.float32).mean(0), np.array(X_train[args.single_client]['data'][:,args.con_idxs],dtype=np.float32).std(0)
        train_std[args.single_client] = np.where(train_std_s < 1e-6, 1e-6, train_std_s)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, ips_dict, mask_dict



def embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model):
    # print("con_mask", con_mask)
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    # print("cate", model.categories_offset)
    x_categ_enc = model.embeds(x_categ)
    # print("cate", x_categ_enc.shape)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    # print("cate", con_mask_temp, con_mask, cat_mask_temp, cat_mask)
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_categ, x_categ_enc, x_cont_enc

def data_split(X, y, nan_mask, ips, rowips, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices],
        'ips': ips.values[indices],
        'rowips': rowips.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d


class DataSetCatCon(data.Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):

        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X_ips = X['ips'].copy()
        rowips = X['rowips'].copy()
        rowips = rowips.squeeze()
        X = X['data'].copy()

        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        self.X1_ips = X_ips[:, cat_cols].copy().astype(np.float32)
        self.X2_ips = X_ips[:, con_cols].copy().astype(np.float32)
        self.rowips = rowips.copy().astype(np.float32)
        if task == 'clf':
            self.y = Y['data']  # .astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        self.cls_ips = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate(
            (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], np.concatenate(
            (self.cls_ips[idx], self.X1_ips[idx])), self.X2_ips[idx], self.rowips[idx]

class DataSetCatCon_mlp(data.Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):

        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X_ips = X['ips'].copy()
        rowips = X['rowips'].copy()
        rowips = rowips.squeeze()
        X = X['data'].copy()

         # categorical columns
        self.X2 = X.copy().astype(np.float32)  # numerical columns

        if task == 'clf':
            self.y = Y['data']  # .astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        self.cls_ips = np.ones_like(self.y, dtype=int)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return self.X2[idx], self.y[idx]
class DatasetFLViT(data.Dataset):
    def __init__(self, args, phase ):
        super(DatasetFLViT, self).__init__()
        self.phase = phase




        if args.dataset == "cifar10" or args.dataset == 'CelebA':
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
            data_all = data_all.item()


            self.data_all = data_all[args.split_type]

            if self.phase == 'train':
                if args.dataset == 'cifar10':
                    self.data = self.data_all['data'][args.single_client]
                    self.labels = self.data_all['target'][args.single_client]
                else:
                    self.data = self.data_all['train'][args.single_client]['x']
                    self.labels = data_all['labels']
            else:
                if args.dataset == 'cifar10':

                    self.data = data_all['union_' + phase]['data']
                    self.labels = data_all['union_' + phase]['target']

                else:
                    if args.split_type == 'real' and phase == 'val':
                        self.data = self.data_all['val'][args.single_client]['x']
                    elif args.split_type == 'central' or phase == 'test':
                        self.data = list(data_all['central']['val'].keys())

                    self.labels = data_all['labels']

        # for Retina dataset
        elif args.dataset =='Retina' :
            if self.phase == 'test':
                args.single_client = os.path.join(args.data_path, 'test.csv')
            elif self.phase == 'val':
                args.single_client = os.path.join(args.data_path, 'val.csv')

            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
            self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

            self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                          open(os.path.join(args.data_path, 'labels.csv'))}

            args.loadSize = 256
            args.fineSize_w = 224
            args.fineSize_h = 224
            self.transform = None


        self.args = args


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.args.dataset == 'cifar10':
            img, target = self.data[index], self.labels[index]
            img = Image.fromarray(img)

        elif self.args.dataset == 'CelebA':
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, 'img_align_celeba', name)
            img = Image.open(path).convert('RGB')
            target = np.asarray(target).astype('int64')
        elif self.args.dataset == 'Retina':
            index = index % len(self.img_paths)

            path = os.path.join(self.args.data_path, 'train-all', self.img_paths[index])
            name = self.img_paths[index]

            ## use new way
            img = np.load(path)
            target = self.labels[name]
            target = np.asarray(target).astype('int64')

            if self.phase == 'train':
                if random.random() < 0.5:  # flip
                    img = np.fliplr(img).copy()
                else:  # not flip
                    img = np.array(img)
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                w_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_w - 1))
                h_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_h - 1))
                img = img[w_offset:w_offset + self.args.fineSize_w, h_offset:h_offset + self.args.fineSize_h]

            else:
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                img = np.array(img)
                img = img[(self.args.loadSize - self.args.fineSize_w) // 2:(self.args.loadSize - self.args.fineSize_w) // 2 + self.args.fineSize_w,
                      (self.args.loadSize - self.args.fineSize_h) // 2:(self.args.loadSize - self.args.fineSize_h) // 2 + self.args.fineSize_h]

            img = torch.from_numpy(img).float()  # send to torch
            img = (1 + 1) / 255 * (img - 255) + 1

            if img.dim() < 3:
                img = torch.stack((img, img, img))
            else:
                img = img.permute(2,1,0)


        if self.transform is not None:
            img = self.transform(img)


        return img,  target


    def __len__(self):
        if self.args.dataset == 'Retina' :
            return len(self.img_paths)

        else:
            return len(self.data)


def random_split(n, num_groups=10):
    numbers = list(range(n))
    random.shuffle(numbers)
    avg_size = n // num_groups
    remainder = n % num_groups

    groups = [numbers[i * avg_size + min(i, remainder):(i + 1) * avg_size + min(i + 1, remainder)] for i in
                      range(num_groups)]

    return groups

def create_dataset_and_evalmetrix(args):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    else:
        datafull = pd.read_csv(
            Path("/data/lsw/data/data/" + args.dataset + "/" + "mcar_" + args.dataset + "_" + str(
                0.0) + ".csv"))
        datamiss = pd.read_csv(Path(
            "/data/lsw/data/data/" + args.dataset + "/" + args.missingtype + args.dataset + "_" + str(
                args.missingrate) + ".csv"))

        n, nfeat = datafull.shape
        args.datafull = datafull
        args.datamiss = datamiss
        args.dis_cvs_files = list(range(args.n_clients)) # 10个客户端
        args.data_all = random_split(n, args.n_clients) # 10个客户端分到的样本即行数
        args.clients_with_len = {name: len(args.data_all[name]) for name in args.dis_cvs_files} # 10个客户端拥有的样本量



    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_auc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []
        args.current_test_auc[single_client] = []
        args.best_eval_loss[single_client] = 9999






