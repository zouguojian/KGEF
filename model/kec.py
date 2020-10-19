# -- coding: utf-8 --
# -- coding: utf-8 --
from load_data import Data
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Experiment:

    def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=100, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

    def get_data_idxs(self, data):  # return 将每个三元组转化为id，如（12，23，45）
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):  # return 头实体，关系，所有邻居 (9186, 7): [3748, 8767, 11974, 7983, 10367]
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):  # return (128, 2) torch.Size([128, 14541]),即头实体，关系, 所有的尾实体
        batch = er_vocab_pairs[idx:min(idx + self.batch_size, len(er_vocab_pairs))]
        targets = np.zeros((len(batch), len(d.entities)+1))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def pad(self, arr, pad_v,max_l=512):
        max_len = min([max([len(line) for line in arr]),max_l])
        for i in range(len(arr)):
            if len(arr[i])>max_len:arr[i]=arr[i][0:max_len]
            else: arr[i] = arr[i] + [pad_v] * (max_len - len(arr[i]))
        return arr

    # def pad(self, arr, pad_v):
    #     max_len = max([len(line) for line in arr])
    #     for i in range(len(arr)):
    #         arr[i] = arr[i] + [pad_v] * (max_len - len(arr[i]))
    #     return arr

    def entity_neighbours_relations(self,data_idxs): #用于获取实体 其关系和邻居实体
        entity_neighbours_relations = defaultdict(list)
        for triple in data_idxs:
            entity_neighbours_relations[triple[0]].append([triple[1],triple[2]])
        return entity_neighbours_relations

    def evaluate(self, model, data,entity_neighbours_relations1,entity_neighbours_relations2):
        hits = []
        ranks = []
        classification=[]
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)  # data_batch: [batch, 3]
            e2_idx = torch.tensor(data_batch[:, 2])

            head_entities1, head_relations1, tail_entities1, tail_relations1, relations = [], [], [], [], []

            # graph1
            for k in range(data_batch.shape[0]):
                head_entities1.append(
                    [data_batch[k][0]] + list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 1].reshape(-1)))
                head_relations1.append(list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 0].reshape(-1)))

                relations.append(data_batch[k][1])
                tail_e = data_batch[k][2]
                if len(entity_neighbours_relations1[tail_e]) > 0:
                    tail_entities1.append(
                        [tail_e] + list(np.array(entity_neighbours_relations1[tail_e])[:, 1].reshape(-1)))
                    tail_relations1.append(list(np.array(entity_neighbours_relations1[tail_e])[:, 0].reshape(-1)))
                else:
                    tail_entities1.append([tail_e])
                    tail_relations1.append([])

            # ([batch_size, neighbours],[batch_size, neighbors-1],[batch_size, neighbours],[batch_size, neighbors-1],[batch_size,relations])
            # alignment graph2
            head_entities2, head_relations2 = [], []
            for k in range(data_batch.shape[0]):
                if data_batch[k][0] in entity_neighbours_relations2:
                    head_entities2.append(
                        [data_batch[k][0]] + list(np.array(entity_neighbours_relations2[data_batch[k][0]])[:, 1].reshape(-1)))
                    head_relations2.append(list(np.array(entity_neighbours_relations2[data_batch[k][0]])[:, 0].reshape(-1)))
                else:
                    head_entities2.append(
                        [data_batch[k][0]] + list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 1].reshape(-1)))
                    head_relations2.append(list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 0].reshape(-1)))

            head_entities1, head_relations1, tail_entities1, tail_relations1, relations = torch.tensor(
                self.pad(head_entities1, len(d.entities))), \
                                                                                          torch.tensor(
                                                                                              self.pad(head_relations1,
                                                                                                       len(
                                                                                                           d.relations))), \
                                                                                          torch.tensor(
                                                                                              self.pad(tail_entities1,
                                                                                                       len(
                                                                                                           d.entities))), \
                                                                                          torch.tensor(
                                                                                              self.pad(tail_relations1,
                                                                                                       len(
                                                                                                           d.relations))), \
                                                                                          torch.tensor(relations)
            head_entities2, head_relations2 = torch.tensor(self.pad(head_entities2, len(d.entities))), torch.tensor(
                self.pad(head_relations2, len(d.relations)))
            if self.cuda:
                e2_idx=e2_idx.cuda()
                head_entities1, head_relations1, tail_entities1, tail_relations1, relations = head_entities1.cuda(), \
                                                                                              head_relations1.cuda(), \
                                                                                              tail_entities1.cuda(), \
                                                                                              tail_relations1.cuda(), \
                                                                                              relations.cuda()
                head_entities2, head_relations2 = head_entities2.cuda(), head_relations2.cuda()

            task1_output, task2_output = model.process(
                triplets1=(head_entities1, head_relations1, tail_entities1, tail_relations1, relations),
                triplets2=(head_entities2, head_relations2))  # 输出预测


            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]  # 所有的尾实体id
                target_value = task1_output[j, e2_idx[j]].item()  # 返回一个值
                task1_output[j, filt] = 0.0
                task1_output[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(task1_output, dim=1, descending=True)  # 将prediction tesor按照指定维度降序排序

            sort_idxs = sort_idxs.cpu().numpy()
            e2_idx = e2_idx.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j])[0][0]
                ranks.append(rank + 1)

                if task2_output[j]>=0.5:classification.append(1.0) #分类任务
                else:classification.append(0.0)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('classification average error: {0}'.format(np.mean(classification)))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):
        print("Training the %s model..." % model_name)
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}  # 给定实体和关系 id
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs1 = self.get_data_idxs(d.train_data1)
        train_data_idxs2 = self.get_data_idxs(d.train_data2)
        print("Number of training data1 points: %d" % len(train_data_idxs1))
        print("Number of training data2 points: %d" % len(train_data_idxs2))

        entity_neighbours_relations1=self.entity_neighbours_relations(self.get_data_idxs(d.data1))
        entity_neighbours_relations2 = self.entity_neighbours_relations(self.get_data_idxs(d.data2))

        model = JointLeatning(d,self.ent_vec_dim,self.rel_vec_dim,**self.kwargs)
        print([value.numel() for value in model.parameters()])

        if self.cuda:
            model.cuda()
        # model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab1 = self.get_er_vocab(train_data_idxs1)
        er_vocab_pairs1 = list(er_vocab1.keys())  # 头实体和关系对

        er_vocab2 = self.get_er_vocab(train_data_idxs2)
        er_vocab_pairs2 = list(er_vocab2.keys())  # 头实体和关系对

        print("Starting training...")

        for it in range(1, self.num_iterations + 1):
            model.train()   #训练
            losses = []
            er_vocab=er_vocab1
            er_vocab_pairs=er_vocab_pairs1

            np.random.shuffle(er_vocab_pairs)

            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets1 = self.get_batch(er_vocab, er_vocab_pairs, j)  # 得到输入值和真实的标签
                opt.zero_grad()  # 清除梯度
                head_entities1,head_relations1, tail_entities1, tail_relations1,relations = [],[],[],[],[]

                #graph1
                for k in range(data_batch.shape[0]):
                    head_entities1.append([data_batch[k][0]]+list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:,1].reshape(-1)))
                    head_relations1.append(list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:,0].reshape(-1)))

                    relations.append(data_batch[k][1])
                    # print(er_vocab[tuple(data_batch[k])])
                    tail_e=er_vocab[tuple(data_batch[k])][random.randint(a=0,b=len(er_vocab[tuple(data_batch[k])])-1)]
                    if len(entity_neighbours_relations1[tail_e])>0:
                        tail_entities1.append([tail_e]+list(np.array(entity_neighbours_relations1[tail_e])[:,1].reshape(-1)))
                        tail_relations1.append(list(np.array(entity_neighbours_relations1[tail_e])[:,0].reshape(-1)))
                    else:
                        tail_entities1.append([tail_e])
                        tail_relations1.append([])
                # ([batch_size, neighbours],[batch_size, neighbors-1],[batch_size, neighbours],[batch_size, neighbors-1],[batch_size,relations])
                # alignment graph2
                head_entities2, head_relations2= [], []
                for k in range(data_batch.shape[0]):
                    if data_batch[k][0] in entity_neighbours_relations2:
                        head_entities2.append([data_batch[k][0]]+list(np.array(entity_neighbours_relations2[data_batch[k][0]])[:, 1].reshape(-1)))
                        head_relations2.append(list(np.array(entity_neighbours_relations2[data_batch[k][0]])[:, 0].reshape(-1)))
                    else:
                        head_entities2.append([data_batch[k][0]]+list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 1].reshape(-1)))
                        head_relations2.append(list(np.array(entity_neighbours_relations1[data_batch[k][0]])[:, 0].reshape(-1)))

                head_entities1, head_relations1, tail_entities1, tail_relations1, relations=torch.tensor(self.pad(head_entities1,len(d.entities))),\
                                                                                            torch.tensor(self.pad(head_relations1,len(d.relations))),\
                                                                                            torch.tensor(self.pad(tail_entities1,len(d.entities))),\
                                                                                            torch.tensor(self.pad(tail_relations1,len(d.relations))),\
                                                                                            torch.tensor(relations)
                head_entities2, head_relations2=torch.tensor(self.pad(head_entities2,len(d.entities))),torch.tensor(self.pad(head_relations2,len(d.relations)))
                if self.cuda:
                    head_entities1, head_relations1, tail_entities1, tail_relations1, relations=head_entities1.cuda(),\
                                                                                                head_relations1.cuda(),\
                                                                                                tail_entities1.cuda(),\
                                                                                                tail_relations1.cuda(),\
                                                                                                relations.cuda()
                    head_entities2, head_relations2=head_entities2.cuda(),head_relations2.cuda()

                task1_output, task2_output = model.process(
                    triplets1=(head_entities1, head_relations1, tail_entities1, tail_relations1, relations),
                    triplets2=(head_entities2,head_relations2))  #输出预测

                # predictions = model.forward(e1_idx, r_idx)  # 输出预测
                if self.label_smoothing:
                    targets1 = ((1.0 - self.label_smoothing) * targets1) + (1.0 / targets1.size(1))
                if self.cuda:
                    loss = torch.add(model.loss(task1_output, targets1),model.r*model.loss(task2_output, torch.ones(size=(data_batch.shape[0],1)).cuda()))
                else:loss = torch.add(model.loss(task1_output, targets1),model.r*model.loss(task2_output, torch.ones(size=(data_batch.shape[0],1))))
                loss.backward()
                opt.step()

                if j==0 or j%(100*self.batch_size)==0:
                    print('In the training procedure, after %d epochs and %d steps, the loss is : %f'%(it,j,loss))
            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            print(it)
            print(np.mean(losses))

            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data,entity_neighbours_relations1,entity_neighbours_relations2)
                if not it % 2:
                    print("Test:")
                    self.evaluate(model, d.test_data,entity_neighbours_relations1,entity_neighbours_relations2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="HypER", nargs="?",
                        help='Which algorithm to use: HypER, ConvE, DistMult, or ComplEx')
    parser.add_argument('--dataset', type=str, default="FB15k-237", nargs="?",
                        help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    args = parser.parse_args()
    model_name = args.algorithm
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = False
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(model_name, num_iterations=800, batch_size=16, learning_rate=0.001,
                            decay_rate=0.99, ent_vec_dim=200, rel_vec_dim=200, cuda=False,
                            input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2,
                            in_channels=1, out_channels=32, filt_h=1, filt_w=9, label_smoothing=0.1)
    experiment.train_and_eval()
