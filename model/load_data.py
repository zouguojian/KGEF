# -- coding: utf-8 --
class Data:

    def __init__(self, data_dir="data/FB15k-237", reverse=False):
        self.train_data1 = self.load_data(data_dir, "train0.4", reverse=reverse)
        self.train_data2 = self.load_data(data_dir, "train0.6", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data1+self.train_data2 + self.valid_data + self.test_data

        self.data1=self.train_data1+self.valid_data + self.test_data
        self.data2 = self.train_data2 + self.valid_data + self.test_data


        self.entities = self.get_entities(self.data)

        self.train_relations1 = self.get_relations(self.train_data1)
        self.train_relations2 = self.get_relations(self.train_data2)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)

        self.relations = self.train_relations1 + [i for i in self.train_relations2 \
                if i not in self.train_relations1]+[i for i in self.valid_relations \
                if i not in self.train_relations1] + [i for i in self.test_relations \
                if i not in self.train_relations1]

    def load_data(self, data_dir, data_type="train", reverse=False): #返回的是所有的三元组
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data): #获取所有的关系
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data): #获取所有的实体
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities