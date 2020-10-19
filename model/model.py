# -- coding: utf-8 --
import torch
import torch.nn as nn
from torch.nn import functional as F,Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_

class Model(nn.Module):  #主要面向知识图谱
    def __init__(self,d,d1=512,d2=512,**kwargs):
        super(Model,self).__init__()
        self.dimension=d1
        self.hidden_size=128
        self.full_layer=512
        self.entity_embedding=nn.Embedding(len(d.entities)+1, d1)
        self.relation_embedding=nn.Embedding(len(d.relations)+1,d2)
        self.full_layer1 = nn.Linear(self.hidden_size, self.full_layer)
        self.task_layer1 = nn.Linear(self.full_layer, self.dimension)
        self.full_layer2 = nn.Linear(self.hidden_size, self.full_layer)
        self.task_layer2 = nn.Linear(self.full_layer, 1)
        self.layer = nn.Linear(self.dimension * 2, self.dimension)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities)+1)))
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])

        self.init()
        self.self_attention()
        self.lstm()

    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)
        # nn.init.normal_(tensor=self.entity_embedding.weight,mean=0.0,std=1.0)
        # nn.init.normal_(tensor=self.relation_embedding.weight,mean=0.0,std=1.0)

    def forward(self):
        return

    def task_special_layers(self,x,char='1'):
        '''
        task special layers!!!
        :param char:
        :return:
        '''
        if char=='1':
            x=F.relu(self.full_layer1(x))
            x = self.hidden_drop(x)
            x = F.relu(self.task_layer1(x))
            x = self.hidden_drop(x)
            x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
            x += self.b.expand_as(x)
            pred = torch.sigmoid(x)
        else:
            x=F.relu(self.full_layer2(x))
            x = self.hidden_drop(x)
            x = F.relu(self.task_layer2(x))
            pred = torch.sigmoid(x)
        return pred

    def concatenation(self,h,r,t):
        return torch.cat(tensors=(h,r,t),dim=1)

    def lstm(self):
        '''
        model layer!!!
        '''
        self.lstm_=nn.LSTM(input_size=self.dimension,hidden_size=128,num_layers=2,dropout=0.1)

    def self_attention(self):
        '''
        self attention!!!
        '''
        encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=1,dim_feedforward=512,dropout=0.1,activation='relu')
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=1)

    def entity_relation_features(self,head_entities=None,head_relations=None, tail_entities=None,tail_relations=None, relations=None):
        '''
        用于提取实体特征
        :param head_entities: [batch_size, neighbours], 邻居也包括自己
        :param head_relations: [batch_size, neighbors-1]
        :param tail_entities: [batch_size, neighbours], 邻居也包括自己
        :param tail_relation: [batch_size, neighbors-1]
        :return:[batch_size, dimension],[batch_size,dimension],[batch_size, dimension]
        '''

        head_entities, tail_entities, head_relations, tail_relations, relations =  self.entity_embedding(head_entities), \
                                                                                   self.entity_embedding(tail_entities), \
                                                                                   self.relation_embedding(head_relations), \
                                                                                   self.relation_embedding(tail_relations),\
                                                                                   self.relation_embedding(relations)

        print(head_entities.shape, tail_entities.shape, head_relations.shape, tail_relations.shape, relations.shape)
        if head_entities.shape[1]!=head_relations.shape[1]:
            head_entities = torch.cat(tensors=(head_entities[:,0:1,:], self.layer(torch.cat(tensors=(head_entities[:,1:,:], head_relations), dim=2))),
                                 dim=1)
        else:
            head_entities = torch.cat(tensors=(
            head_entities[:, 0:1, :], self.layer(torch.cat(tensors=(head_entities[:, 1:, :], head_relations[:,:head_relations.shape[1]-1,:]), dim=2))),
                                      dim=1)
        if tail_entities.shape[1]!=tail_relations.shape[1]:
            tail_entities = torch.cat(tensors=(tail_entities[:,0:1,:], self.layer(torch.cat(tensors=(tail_entities[:,1:,:], tail_relations), dim=2))),
                                 dim=1)
        else:
            tail_entities = torch.cat(tensors=(
            tail_entities[:, 0:1, :], self.layer(torch.cat(tensors=(tail_entities[:, 1:, :], tail_relations[:,:tail_relations.shape[1]-1,:]), dim=2))),
                                      dim=1)
        head_entities= self.transformer_encoder(head_entities)[:,0,:]
        tail_entities = self.transformer_encoder(tail_entities)[:,0,:]

        return head_entities, relations, tail_entities

class JointLeatning(nn.Module):#主要面向联合学习
    def __init__(self,d, d1, d2,**kwargs):
        super(JointLeatning, self).__init__()
        self.learning_rate=0.001
        self.model=Model(d, d1, d2,**kwargs)
        self.sent_len=512
        self.r=torch.tensor(1.0)
        self.loss = torch.nn.BCELoss()
        self.layer1 = nn.Linear(self.model.dimension, 1)
        self.layer2 = nn.Linear(self.model.dimension, 1)
        self.layer_norm=torch.nn.LayerNorm(normalized_shape=[self.model.dimension])

    def feature_extact(self,x): #特征提取和任务层
        '''
        :param x: [head_entities, relations, tail_entities]
        :return:
        '''

        output, (h, c)=self.model.lstm_(x) # output: [batch, time_size, hidden_size],h,c: [batch_size, 1 ,hidden_size]
        output1=self.model.task_special_layers(output[:,-2,:],'1')     #task layer
        output2 = self.model.task_special_layers(output[:, -1,:], '2')
        return output1, output2

    def norm(self,x):
        return self.layer_norm(x)

    def cross_matrix(self,entity1=None,entity2=None):
        '''
        :param entity1:
        :param entity2:
        :return:
        '''
        entity1=entity1.unsqueeze(dim=2)
        entity2=entity2.unsqueeze(dim=1)
        entity1_out=F.relu(self.layer1(torch.matmul(entity1,entity2)))
        entity2_out=F.relu(self.layer1(torch.matmul(entity2.permute(dims=[0,2,1]),entity1.permute(dims=[0,2,1]))))
        entity1=self.norm(torch.squeeze(entity1_out))
        entity2=self.norm(torch.squeeze(entity2_out))
        return torch.add(entity1,entity2)

    def prediction(self, head_entities1, relations1, tail_entities1):

        head_entities1, relations1, tail_entities1 = torch.unsqueeze(head_entities1, dim=1), \
                                                     torch.unsqueeze(relations1, dim=1), \
                                                     torch.unsqueeze(tail_entities1, dim=1)

        input_triplets = self.model.concatenation(head_entities1, relations1,
                                                  tail_entities1)  # concate the input date before input to the lstm model
        task1_output, taks2_output = self.feature_extact(input_triplets)

        return task1_output, taks2_output

    def process(self,triplets1=None, triplets2=None):
        '''
        :param triplets1: ([batch_size, neighbours],[batch_size, neighbors-1],[batch_size, neighbours],[batch_size, neighbors-1],[batch_size,relations])
        :param triplets2: ([batch_size, neighbours],[batch_size, neighbors-1])
        :param is_alignment: 实体是否对齐，若没有对齐，则不需要交叉
        :return:
        '''

        head_entities1, relations1, tail_entities1=self.model.entity_relation_features(head_entities=triplets1[0],
                                                                                       head_relations=triplets1[1],
                                                                                       tail_entities=triplets1[2],
                                                                                       tail_relations=triplets1[3],
                                                                                       relations=triplets1[4])

        head_entities2, head_relations2= self.model.entity_embedding(triplets2[0]), self.model.relation_embedding(triplets2[1])
        if head_entities2.shape[1]!=head_relations2.shape[1]:
            head_entities2 = torch.cat(tensors=(head_entities2[:, 0:1, :],
                                                self.model.layer(torch.cat(tensors=(head_entities2[:, 1:, :], head_relations2), dim=2))),
                                      dim=1)
        else:
            head_entities2 = torch.cat(tensors=(head_entities2[:, 0:1, :],
                                                self.model.layer(torch.cat(tensors=(head_entities2[:, 1:, :], head_relations2[:,:head_relations2.shape[1]-1,:]), dim=2))),
                                      dim=1)
        head_entities2 = self.model.transformer_encoder(head_entities2)[:, 0, :]

        e=self.cross_matrix(head_entities1,head_entities2)
        head_entities1=e

        return self.prediction(head_entities1, relations1, tail_entities1)



# class Experiment:
#
#     def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
#                  num_iterations=100, batch_size=128, decay_rate=0., cuda=False,
#                  input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
#                  in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
#         self.model_name = model_name
#         self.learning_rate = learning_rate
#         self.ent_vec_dim = ent_vec_dim
#         self.rel_vec_dim = rel_vec_dim
#         self.num_iterations = num_iterations
#         self.batch_size = batch_size
#         self.decay_rate = decay_rate
#         self.label_smoothing = label_smoothing
#         self.cuda = cuda
#         self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
#                        "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
#                        "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

# joit_learning=JointLeatning()
# #([batch_size, neighbours],[batch_size, neighbors-1],[batch_size, neighbours],[batch_size, neighbors-1],[batch_size,relations])
# head_entities=torch.randint(low=0,high=50,size=[10,32])
# head_relations=torch.randint(low=0,high=99,size=[10,31])
#
# tail_entities=torch.randint(low=0,high=50,size=[10,12])
# tail_relations=torch.randint(low=0,high=99,size=[10,11])
#
# relations=torch.randint(low=0,high=99,size=[10])
#
# task1_output,taks2_output=joit_learning.process(triplets1=(head_entities,head_relations,tail_entities,tail_relations,relations),
#                                                 triplets2=(head_entities,head_relations))
#
# print(joit_learning.model.entity_embedding.weight.shape)
# print(task1_output.shape,taks2_output.shape)
# print(task1_output,taks2_output)