import os
import re
import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from random import gauss
from sklearn.impute import KNNImputer
import torch
from torch_geometric.data import Data

def get_dict(data):
    change = data.drop_duplicates().to_list()
    change_dict = {}
    for i in range(len(change)):
        change_dict[change[i]] = i
    return change, change_dict

def is_defect(defect_ratio):
    #随机不确定缺损
    #三种组合缺损情况概率和为defect_ratio,具体缺损组合不确定
    #维持原状不缺损概率为1-defect_ratio
    a=random.uniform(0, defect_ratio)
    b=random.uniform(0, defect_ratio-a)
    c=defect_ratio-a-b
    case = np.random.choice([0,1,2,3], p=[a,b,c,1-defect_ratio])
    return case

def pre_process_KuaiRec_data():
    print("Loading KuaiRec...")

    # user_item
    data = pd.read_csv("DATA/Origin_KuaiRec/big_matrix.csv")
    user_item = data[['user_id', 'video_id']].drop_duplicates().reset_index(drop=True)
    user_item.to_csv('DATA/Origin_KuaiRec/processed_user_item.csv', sep=',', index=False, encoding='utf-8')
    print('Generate files, processed_user_item.csv')

    # item_feature
    item_daily_features = pd.read_csv("DATA/Origin_KuaiRec/item_daily_features.csv")
    item_features = item_daily_features[
        ['video_id', 'author_id', 'video_type', 'upload_type', 'visible_status', 'video_duration', 'video_width',
         'video_height', 'music_id', 'video_tag_id']]
    item_features = item_features.fillna(0).drop_duplicates().reset_index(drop=True)  # 去重,行索引重排
    # 替换
    _, music_id_dict = get_dict(item_features['music_id'])
    item_features['music_id'] = item_features['music_id'].map(music_id_dict)
    _, video_type_dict = get_dict(item_features['video_type'])
    item_features['video_type'] = item_features['video_type'].map(video_type_dict)
    _, upload_type_dict = get_dict(item_features['upload_type'])
    item_features['upload_type'] = item_features['upload_type'].map(upload_type_dict)
    _, visible_status_dict = get_dict(item_features['visible_status'])
    item_features['visible_status'] = item_features['visible_status'].map(visible_status_dict)
    # 合并
    tag = item_features.groupby(['video_id'])['video_tag_id'].apply(list).apply(lambda x: str(x).replace(',', ''))
    item_features = item_features.drop_duplicates(subset='video_id').reset_index(drop=True)
    item_features['video_tag_id'] = tag
    # 保存
    item_features.to_csv('DATA/Origin_KuaiRec/processed_item_features.csv', sep=',', index=False, encoding='utf-8')
    print('Generate files, processed_item_features.csv')

    # user_feature
    user_features = pd.read_csv("DATA/Origin_KuaiRec/user_features.csv")
    dict_1 = {'UNKNOWN': 0, 'middle_active': 1, 'high_active': 2, 'full_active': 3}
    dict_6 = {'0': 0, '(0,10]': 1, '(10,50]': 2, '(50,100]': 3, '(100,150]': 4, '(150,250]': 5, '(250,500]': 6,
              '500+': 7}
    dict_8 = {'0': 0, '[1,10)': 1, '[10,100)': 2, '[100,1k)': 3, '[1k,5k)': 4, '[5k,1w)': 5, '[1w,10w)': 6}
    dict_10 = {'0': 0, '[1,5)': 1, '[5,30)': 2, '[30,60)': 3, '[60,120)': 4, '[120,250)': 5, '250+': 6}
    dict_12 = {'15-30': 0, '31-60': 1, '61-90': 2, '91-180': 3, '181-365': 4, '366-730': 5, '730+': 6}
    user_features['user_active_degree'] = user_features['user_active_degree'].map(dict_1)
    user_features['follow_user_num_range'] = user_features['follow_user_num_range'].map(dict_6)
    user_features['fans_user_num_range'] = user_features['fans_user_num_range'].map(dict_8)
    user_features['friend_user_num_range'] = user_features['friend_user_num_range'].map(dict_10)
    user_features['register_days_range'] = user_features['register_days_range'].map(dict_12)
    user_features = user_features.fillna(0)
    user_features.to_csv('DATA/Origin_KuaiRec/processed_user_features.csv', sep=',', index=False, encoding='utf-8')
    print('Generate files, processed_user_features.csv')
    print("Pre process KuaiRec finish")

def pre_process_TenRec_data():
    print("Loading TenRec...")
    data = pd.read_csv("DATA/Origin_TenRec/cold_data.csv")  # (343305, 20)
    user_original, user_id_dict = get_dict(data['user_id'])  # 68661user 55746item
    data['user_id'] = data['user_id'].map(user_id_dict)
    item_original, item_id_dict = get_dict(data['item_id'])
    data['item_id'] = data['item_id'].map(item_id_dict)
    user_item = data[['user_id', 'item_id']].drop_duplicates().reset_index(drop=True)
    user_item.to_csv('DATA/Origin_TenRec/processed_user_item.csv', sep=',', index=False, encoding='utf-8')
    #user_features
    user_features = data[['user_id', 'gender', 'age']].drop_duplicates(subset=['user_id']).reset_index(drop=True)
    user_features.to_csv('DATA/Origin_TenRec/processed_user_features.csv', sep=',', index=False, encoding='utf-8')
    #item_features
    item_features = data[
        ['item_id', 'click_count', 'like_count', 'comment_count', 'item_score1', 'item_score2', 'item_score3',
         'category_first', 'category_second']].drop_duplicates(subset=['item_id']).reset_index(drop=True)
    item_features.to_csv('DATA/Origin_TenRec/processed_item_features.csv', sep=',', index=False, encoding='utf-8')
    #user-user
    user_item['item_id'] += 68661
    temp_user_user_weight={}
    group = user_item.groupby('item_id')
    for item_id, users_item in group:
        users=users_item['user_id'].tolist()
        if(len(users)>1):
            #每个user分到1/len(users),user-user为1/len(users)+1/len(users)
            relations = list(itertools.combinations([a for a in users], 2))
            for user_user in relations:
                if user_user not in temp_user_user_weight:
                    temp_user_user_weight[user_user] = 1/len(users)*2
                else:
                    temp_user_user_weight[user_user] +=1/len(users)*2
    user_user_weight={k: v for k, v in temp_user_user_weight.items() if v >= 1}
    print(len(temp_user_user_weight))
    print(len(user_user_weight))
    user_user_pd=pd.DataFrame(user_user_weight.keys(), columns=('user0', 'user1'))
    user_user_pd.to_csv('DATA/Origin_TenRec/processed_user_user.csv', sep=',', index=False, encoding='utf-8')
    print("Pre process TenRec finish")

def missing_value(x,completion): #缺失值处理
    print('Missing value completion method: ',completion)
    if completion=='zero':
        # 法1:缺失值用0补全
        x[np.isnan(x)] = 0
    elif completion=='mean':
        # 法2:缺失值均值补全
        col_mean = np.nanmean(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])
    elif completion=='knn':
        # 法3:缺失值KNN补全
        imputer = KNNImputer(n_neighbors=2)
        x = imputer.fit_transform(x)
    elif completion=='gauss':
        # 法3:缺失值高斯分布随机生成数补全
        x[np.isnan(x)] = [gauss(0, 1) for i in range(len(x[np.isnan(x)]))]
    return x

def load_KuaiRec_data(completion='zero',defect_ratio=0.3):
    print('Loading KuaiRec ...')
    data_x_file = 'DATA/KuaiRec/data_x.pt'
    edge_index_file = 'DATA/KuaiRec/edge_index.pt'
    kuairec_data_file = 'DATA/KuaiRec/data.csv'
    if os.path.exists(data_x_file) and os.path.exists(edge_index_file) and os.path.exists(kuairec_data_file):
        # 相关数据集已存在
        x = torch.load(data_x_file)
        edge_index = torch.load(edge_index_file)
        gcn_data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([0]))
        kuairec_data = pd.read_csv(kuairec_data_file)
    else:
        # 采样
        yes_edges = []
        no_edges = []
        yes_y = []
        no_y = []
        # user-user
        social_network = pd.read_csv("DATA/Origin_KuaiRec/social_network.csv")
        for i in range(social_network.shape[0]):
            user0 = social_network.loc[i]['user_id']
            all_users = [i for i in range(7176)]
            all_users.remove(user0)
            user1s = social_network.loc[i]['friend_list']
            results = re.findall('([0-9]+)', user1s)
            yeslist = [int(result) for result in results]
            nolist = random.sample(list(filter(lambda x: x not in yeslist, all_users)), len(yeslist))
            yes_edges += [(user0, user1) for user1 in yeslist]
            no_edges += [(user0, user1) for user1 in nolist]
            yes_y += [1] * len(yeslist)
            no_y += [0] * len(nolist)
        # print(len(yes_edges))
        # print(len(no_edges))
        # print(len(yes_y))
        # print(len(no_y))

        # user-item
        data = pd.read_csv("DATA/Origin_KuaiRec/processed_user_item.csv")
        data['video_id'] += 7176  # id范围0-7175为user,7176-17903为item
        group = data.groupby('user_id')
        all_items = [i for i in range(7176, 17904)]
        for user_id, user_items in group:
            yeslist = user_items['video_id'].tolist()
            nolist = random.sample(list(filter(lambda x: x not in yeslist, all_items)), len(yeslist))
            yes_edges += [(user_id, item) for item in yeslist]
            no_edges += [(user_id, item) for item in nolist]
            yes_y += [1] * len(yeslist)
            no_y += [0] * len(nolist)
        # print(len(yes_edges))
        # print(len(no_edges))
        # print(len(yes_y))
        # print(len(no_y))

        # 邻接矩阵
        g = nx.Graph(yes_edges)  # 交互关系转换为图
        nodelist = [i for i in range(17904)]
        adj = nx.to_scipy_sparse_matrix(g, nodelist=nodelist, dtype=int, format='coo')  # 生成图的邻接矩阵的稀疏矩阵
        edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))  # 我们真正需要的coo形式
        x = []
        node_modalitys = {}
        user_features = pd.read_csv("DATA/Origin_KuaiRec/processed_user_features.csv")
        item_features = pd.read_csv("DATA/Origin_KuaiRec/processed_item_features.csv")
        # 节点及节点特征
        for node in nodelist:
            if node < 7176:  # user_node
                node_infor = user_features.loc[node]
                feat = node_infor.tolist()[1:4] + [np.nan, np.nan, np.nan, np.nan, np.nan]
                node_modalitys[node] = 0
            else:  # item_node
                node_infor = item_features.loc[node - 7176]
                case = is_defect(defect_ratio)  # 随机不确定缺损
                if case == 0:  # case=0只有类型状态,设定背景音乐、文本标签缺损
                    feat = node_infor.tolist()[1:4] + [np.nan, np.nan, np.nan, np.nan, np.nan]
                    node_modalitys[node] = 0
                else:
                    music_id = node_infor['music_id']
                    if music_id > 0:
                        m_exist = 1
                        feat_m = [music_id]
                    else:
                        m_exist = 0
                        feat_m = [np.nan]
                    if case == 1:  # case=1类型状态+背景音乐组合,设定文本标签缺损
                        feat = node_infor.tolist()[1:4] + feat_m + [np.nan, np.nan, np.nan, np.nan]
                        node_modalitys[node] = m_exist
                    else:
                        tags = node_infor['video_tag_id']
                        results = re.findall('([0-9]+)', tags)
                        if '0' in results or '-124' in results:  # 设定该模态全部信息缺失
                            t_exist = 0
                            feat_t = [np.nan, np.nan, np.nan, np.nan]
                        else:
                            t_exist = 1
                            feat_t = [int(result) for result in results]
                            feat_t = feat_t[:4]
                            feat_t.extend(0 for _ in range(4 - len(feat_t)))
                        if case == 3:  # case=3表示不增加缺损,保持原有状态
                            feat = node_infor.tolist()[1:4] + feat_m + feat_t
                            node_modalitys[node] = m_exist + t_exist * 2
                            # 3表示类型状态+背景音乐+文本标签组合，2表示类型状态+文本标签组合，1表示类型状态+背景音乐组合，0表示只有类型状态
                        else:  # case==2 类型状态+文本标签组合,设定背景音乐缺损
                            feat = node_infor.tolist()[1:4] + [np.nan] + feat_t
                            node_modalitys[node] = t_exist * 2
            x.append(feat)
        x = np.array(x)
        x = missing_value(x, completion)  # 缺失值补全
        x = torch.FloatTensor(x)
        gcn_data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([0]))
        torch.save(x, data_x_file)
        torch.save(edge_index, edge_index_file)

        edges = yes_edges + no_edges
        y = yes_y + no_y
        kuairec_data = pd.DataFrame(edges, columns=('node0', 'node1'))
        kuairec_data['link_label'] = y
        kuairec_data['type_label'] = (kuairec_data['node1'] >= 7176).astype(int)  # node1>=7176为user-item,否则为user-user
        kuairec_data['modality_label'] = kuairec_data['node1'].map(node_modalitys)  # node0的模态+node1的模态,node0始终是user,模态为0
        kuairec_data.to_csv(kuairec_data_file, sep=',', index=False, encoding='utf-8')
        print('Get KuaiRec data')
    print('Load KuaiRec finish')
    return gcn_data, kuairec_data

def load_TenRec_data(completion='zero',defect_ratio=0.3):
    print("Loading TenRec ...")
    data_x_file = 'DATA/TenRec/data_x.pt'
    edge_index_file = 'DATA/TenRec/edge_index.pt'
    tenrec_data_file = 'DATA/TenRec/data.csv'
    if os.path.exists(data_x_file) and os.path.exists(edge_index_file) and os.path.exists(tenrec_data_file):
        # 相关数据集已存在
        x = torch.load(data_x_file)
        edge_index = torch.load(edge_index_file)
        gcn_data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([0]))
        tenrec_data = pd.read_csv(tenrec_data_file)
    else:
        # 采样
        yes_edges = []
        no_edges = []
        yes_y = []
        no_y = []
        #user-user
        user_user_pd= pd.read_csv("DATA/Origin_TenRec/processed_user_user.csv")
        group = user_user_pd.groupby('user0')
        for user0, relations in group:
            all_users = [i for i in range(68661)]
            all_users.remove(user0)
            yeslist = relations['user1'].tolist()
            nolist = random.sample(list(filter(lambda x: x not in yeslist, all_users)), len(yeslist))
            yes_edges += [(user0, user1) for user1 in yeslist]
            no_edges += [(user0, user1) for user1 in nolist]
            yes_y += [1] * len(yeslist)
            no_y += [0] * len(nolist)
        # print(len(yes_edges))
        # print(len(no_edges))
        # print(len(yes_y))
        # print(len(no_y))

        # user-item
        data = pd.read_csv("DATA/Origin_TenRec/processed_user_item.csv")  #68661user 55746item
        data['item_id'] += 68661  # id范围0-68660为user,68661-124406为item
        group = data.groupby('user_id')
        all_items = [i for i in range(68661, 124407)]
        for user_id, user_items in group:
            yeslist = user_items['item_id'].tolist()
            nolist = random.sample(list(filter(lambda x: x not in yeslist, all_items)), len(yeslist))
            yes_edges += [(user_id, item) for item in yeslist]
            no_edges += [(user_id, item) for item in nolist]
            yes_y += [1] * len(yeslist)
            no_y += [0] * len(nolist)
        # print(len(yes_edges))
        # print(len(no_edges))
        # print(len(yes_y))
        # print(len(no_y))

        # 邻接矩阵
        g = nx.Graph(yes_edges)  # 交互关系转换为图
        nodelist = [i for i in range(124407)]
        adj = nx.to_scipy_sparse_matrix(g, nodelist=nodelist, dtype=int, format='coo')  # 生成图的邻接矩阵的稀疏矩阵
        edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))  # 我们真正需要的coo形式
        x = []
        node_modalitys = {}
        user_features = pd.read_csv("DATA/Origin_TenRec/processed_user_features.csv")
        item_features = pd.read_csv("DATA/Origin_TenRec/processed_item_features.csv")
        # 节点及节点特征
        for node in nodelist:
            if node < 68661:  # user_node
                node_infor = user_features.loc[node].tolist()
                feat = node_infor[1:] + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                node_modalitys[node] = 0
            else:  # item_node
                node_infor = item_features.loc[node - 68661].tolist()
                case = is_defect(defect_ratio)  # 随机不确定缺损
                if case == 0:  # case=0 counts+scores组合,设定categorys缺损
                    node_infor[7:] = [np.nan, np.nan]
                elif case == 1:  # case=1 counts+categorys组合,设定scores缺损
                    node_infor[4:7] = [np.nan, np.nan, np.nan]
                elif case == 2:  # case=2 scores+categorys组合,设定counts缺损
                    node_infor[1:4] = [np.nan, np.nan, np.nan]
                # case=3 counts+scores+categorys组合
                feat = node_infor[1:]
                node_modalitys[node] = case
            x.append(feat)
        x = np.array(x)
        x = missing_value(x, completion)  # 缺失值补全
        x = torch.FloatTensor(x)
        gcn_data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([0]))
        torch.save(x, data_x_file)
        torch.save(edge_index, edge_index_file)

        edges = yes_edges + no_edges
        y = yes_y + no_y
        tenrec_data = pd.DataFrame(edges, columns=('node0', 'node1'))
        tenrec_data['link_label'] = y
        tenrec_data['type_label'] = (tenrec_data['node1'] >= 68661).astype(int)  # node1>=68661为user-item,否则为user-user
        tenrec_data['modality_label'] = tenrec_data['node1'].map(node_modalitys)  # node0的模态+node1的模态,node0始终是user,模态为0
        tenrec_data.to_csv(tenrec_data_file, sep=',', index=False, encoding='utf-8')
        print('Get TenRec data')
    print('Load TenRec finish')
    return gcn_data, tenrec_data

if __name__ == '__main__':
    pre_process_KuaiRec_data()
    pre_process_TenRec_data()

    # load_KuaiRec_data()
    # load_TenRec_data()

