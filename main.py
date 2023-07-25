import os
import shutil
#import pyttsx3
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from data_process import load_KuaiRec_data,load_TenRec_data
import datetime
import warnings
warnings.filterwarnings("ignore")


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(8, hidden_channels)  # 输入通道的大小和节点的特征维度一致
        self.conv2 = GCNConv(hidden_channels, 16)  # 输出通道的大小和节点的特征维度一致,即获取16维节点表示

    def forward(self, x, edge_index):  # 前向传播
        x = self.conv1(x, edge_index)  # 输入为节点及特征和边的稀疏矩阵,输出维度[所有节点数,hidden_channels]
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # 输入为节点及特征和边的稀疏矩阵,输出维度[所有节点数,16]
        return x

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class Adversarial(nn.Module):
    def __init__(self, link_numbers=2, type_numbers=2, modality_numbers=4):
        super(Adversarial, self).__init__()

        self.node_gcn = GCN(hidden_channels=32)

        self.bn = nn.BatchNorm1d(16)

        self.link_classifier = nn.Sequential(
            nn.Linear(32, link_numbers),
            nn.Softmax(dim=1))

        self.type_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, type_numbers),
            nn.Softmax(dim=1))

        self.modality_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, modality_numbers),
            nn.Softmax(dim=1))

    def forward(self,gcn_data, edges, coeff=1):
        x=self.node_gcn(gcn_data.x,gcn_data.edge_index)
        x = self.bn(x)
        edges = torch.split(edges, 1, dim=1)
        node0 = torch.index_select(x, 0, torch.squeeze(edges[0]))
        node1 = torch.index_select(x, 0, torch.squeeze(edges[1]))
        edges = torch.cat((node0, node1), dim=1)
        link_outputs = self.link_classifier(edges)
        type_reverse_feature = grad_reverse(edges, coeff)
        type_outputs = self.type_classifier(type_reverse_feature)
        modality_reverse_features = grad_reverse(edges, coeff)
        modality_outputs = self.modality_classifier(modality_reverse_features)
        return link_outputs, type_outputs, modality_outputs

def train_Adversarial_Model(gcn_data, train_loader, valid_loader, model, criterion,initial_learning_rate,epochs,record_file):
    model_path = 'result/model/'
    if os.path.exists(model_path): # 清除之前运行代码生成的模型
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print(datetime.datetime.now())
    print('Training...')
    best_valid_dir = ''
    best_valid_acc = 0
    for epoch in range(epochs + 1):
        p = epoch / epochs
        learning_rate = initial_learning_rate / pow((1 + 10 * p), 0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        # 测试集
        model.train()
        link_loss_vec = []
        type_loss_vec = []
        modality_loss_vec = []
        loss_vec = []
        acc_vec = []
        for data in train_loader:
            # print(datetime.datetime.now()) #每batch数据的时间
            edges, link_labels, type_labels, modality_labels = data
            if torch.cuda.is_available():
                edges = edges.cuda()
                link_labels = link_labels.cuda()
                type_labels = type_labels.cuda()
                modality_labels = modality_labels.cuda()
            edges = Variable(edges)
            link_labels = Variable(link_labels)
            type_labels = Variable(type_labels)
            modality_labels = Variable(modality_labels)
            # 向前传播
            link_outs, type_outs, modality_outs = model(gcn_data, edges)
            link_loss = criterion(link_outs, link_labels)
            type_loss = criterion(type_outs, type_labels)
            modality_loss = criterion(modality_outs, modality_labels)
            batch_loss = link_loss + type_loss + modality_loss
            loss_vec.append(batch_loss.cpu().detach().numpy())
            link_loss_vec.append(link_loss.cpu().detach().numpy())
            type_loss_vec.append(type_loss.cpu().detach().numpy())
            modality_loss_vec.append(modality_loss.cpu().detach().numpy())

            _, argmax = torch.max(link_outs, 1)
            batch_acc = (argmax == link_labels).float().mean()
            acc_vec.append(batch_acc.item())

            optimizer.zero_grad() #清空过往梯度
            batch_loss.backward() #反向传播,计算当前梯度 retain_graph=True梯度保存
            optimizer.step() #梯度下降,更新网络参数

        link_loss = np.mean(link_loss_vec)
        type_loss = np.mean(type_loss_vec)
        modality_loss = np.mean(modality_loss_vec)
        loss = np.mean(loss_vec)
        acc = np.mean(acc_vec)

        # 验证集
        model.eval()
        valid_acc_vec = []
        for data in valid_loader:
            edges,link_labels, _, _ = data
            if torch.cuda.is_available():
                with torch.no_grad():  # 不计算参数梯度
                    edges = Variable(edges).cuda()
                    link_labels = Variable(link_labels).cuda()
            else:
                with torch.no_grad():
                    edges = Variable(edges)
                    link_labels = Variable(link_labels)
            link_outs, _, _ = model(gcn_data, edges)
            _, argmax = torch.max(link_outs, 1)
            batch_acc = (argmax == link_labels).float().mean()
            valid_acc_vec.append(batch_acc.item())

        # 保存最好的Adversarial模型
        valid_acc = np.mean(valid_acc_vec)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_dir = model_path + 'model' + str(epoch) + '.pkl'
            torch.save(model.state_dict(), best_valid_dir)

        if epoch % 1 == 0:
            print('epoch: [{}/{}], lr:{:.6f}, train loss:{:.6f}, train acc:{:.4f}, valid acc:{:.4f}, l loss:{:.6f}, t loss:{:.6f}, m loss:{:.6f}'.format(epoch, epochs, learning_rate, loss, acc, valid_acc, link_loss, type_loss, modality_loss))
            w = 'Model epoch: [' + str(epoch) + '/' + str(epochs) + '], lr: ' + str(round(learning_rate, 6)) + ', train loss: ' + str(round(loss, 4)) + ', train acc: ' + str(round(acc, 4)) + ', valid acc: ' + str(round(valid_acc, 4)) + ', link loss: ' + str(round(link_loss, 4)) + ', type loss: ' + str(round(type_loss, 4)) + ', modality loss: ' + str(round(modality_loss, 4)) + '\n'
            record_file.write(w)

    return best_valid_dir

def test_Adversarial_Model(gcn_data, test_loader, adversarial_model, best_valid_dir,record_file):
    # 加载最好的Adversarial模型
    adversarial_model.load_state_dict(torch.load(best_valid_dir))
    adversarial_model.eval()
    print(datetime.datetime.now())
    print('Test...')
    scores = []
    labels = []
    preds = []
    # 测试集
    for data in test_loader:
        edges, link_labels, _, _ = data
        if torch.cuda.is_available():
            with torch.no_grad():  # 不计算参数梯度
                edges = Variable(edges).cuda()
                link_labels = Variable(link_labels).cuda()
        else:
            with torch.no_grad():
                edges = Variable(edges)
                link_labels = Variable(link_labels)

        link_outs, _, _ = adversarial_model(gcn_data, edges)
        _, test_argmax = torch.max(link_outs, 1)
        scores += list(link_outs[:, 1].cpu().detach().numpy())
        labels += list(link_labels.cpu().detach().numpy())
        preds += list(test_argmax.cpu().detach().numpy())

    acc = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, average='macro')
    auc = metrics.roc_auc_score(labels, scores, average='macro')
    #auc降低非真样本呈阳性的比例(假阳性),尽量不误报,倾向保守估计
    print("Test acc:{:.4f}, precision:{:.4f}, auc:{:.4f} ".format(acc, precision,auc))
    w = 'Test acc: ' + str(round(acc,4)) + ', precision: ' + str(round(precision,4)) + ', auc: ' + str(round(auc,4)) + '\n'
    record_file.write(w)
    record_file.write('===========================================================================\n')


def run_Adversarial_model(gcn_data, train_loader, valid_loader, test_loader,initial_learning_rate,epochs,record_file):
    adversarial_model = Adversarial()
    if torch.cuda.is_available():
        adversarial_model = adversarial_model.cuda()
    criterion = nn.CrossEntropyLoss()
    best_valid_dir = train_Adversarial_Model(gcn_data, train_loader, valid_loader, adversarial_model, criterion,initial_learning_rate,epochs,record_file)
    test_Adversarial_Model(gcn_data, test_loader, adversarial_model, best_valid_dir,record_file)

def get_loader(data,batch_size):
    data = pd.DataFrame(data, columns=('node0','node1','link_label','type_label','modality_label','split_label'))
    print("edges counter: ", sorted(Counter(list(data['link_label'])).items()))
    edges = torch.LongTensor(np.array(data[['node0','node1']]))
    link_labels = torch.LongTensor(list(data['link_label']))
    type_labels = torch.LongTensor(list(data['type_label']))
    modality_labels = torch.LongTensor(list(data['modality_label']))
    data_set = TensorDataset(edges, link_labels, type_labels, modality_labels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def get_train_vaild_test(data,batch_size):
    data['split_label']=data['link_label']*4+data['type_label']*2+data['modality_label'] #用来成比例划分数据
    print("data counter: ", sorted(Counter(list(data['link_label'])).items()))
    data = shuffle(data).reset_index(drop=True) #数据打乱重排
    labels = list(data['split_label']) #转换类型
    # 7:1:2随机划分
    train_data, test_data, train_labels, test_labels = train_test_split(np.array(data), labels, test_size=0.2)
    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.125)
    train_loader = get_loader(train_data,batch_size)
    valid_loader = get_loader(valid_data,batch_size)
    test_loader = get_loader(test_data,batch_size)
    return train_loader, valid_loader, test_loader

def run_main(dataset):
    print('The program starts running.')
    # 定义超参数:
    initial_learning_rate = 0.01  # 初始学习率
    epochs = 10  # 训练迭代次数
    set_defect_ratio=0.3
    if dataset == 'KuaiRec':
        gcn_data, data = load_KuaiRec_data(completion='mean',defect_ratio=set_defect_ratio)
        batch_size = 8192
    else: #默认'TenRec'
        gcn_data, data = load_TenRec_data(completion='mean',defect_ratio=set_defect_ratio)
        batch_size = 256
    record_file = open('result/out.txt', 'a', encoding='utf-8')
    print('dataset: '+dataset)
    print('defect_ratio: ' + str(set_defect_ratio))
    record_file.write('Dataset: '+dataset + '\n')
    record_file.write('defect_ratio: ' + str(set_defect_ratio) + '\n')
    print('batch:{}, initial lr:{}, epochs:{}'.format(batch_size, initial_learning_rate, epochs))
    record_file.write('batch: ' + str(batch_size) + ', lr: ' + str(initial_learning_rate) + ', epochs: ' + str(epochs) + '\n')

    if torch.cuda.is_available():
        gcn_data.x = gcn_data.x.cuda()
        gcn_data.edge_index = gcn_data.edge_index.cuda()
    train_loader, valid_loader, test_loader = get_train_vaild_test(data,batch_size)
    run_Adversarial_model(gcn_data, train_loader, valid_loader, test_loader,initial_learning_rate,epochs,record_file)
    record_file.close()

if __name__ == '__main__':
    datasets=['KuaiRec','TenRec']
    for repeat in range(5):
        for dataset in datasets:
            begin = datetime.datetime.now()
            print('===========  start time ', begin)
            run_main(dataset)
            end = datetime.datetime.now()
            print('===========  end time ',end)
            print('===========  run time ',end-begin)
