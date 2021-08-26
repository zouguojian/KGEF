# KGEF
Knowledge graph embedding

Knowledge graphs are useful resources for numerous artificial intelligence applications, but they are not complete solutions. The biggest challenge and limitation at present are the data sparsity of a single knowledge graph. In this study, we propose a method for knowledge graph embedding (KGE) based on federated learning (KGEF). This method can solve the problem of multi-source complementary knowledge graph entity security, and it fully considers the diversity of entities and relations to solve the problem of data sparsity. Specifically, federated learning solves the security problem of multi-source knowledge graph entity alignment with data kept locally, and the information interaction between knowledge graphs is completed after homomorphic encryption. In our proposed KGEF method, we use a graph attention network (GAT) to encode the graph as well as to select important neighbors and relations through the attention mechanism, and cross-fuse the encoded encrypted feature vectors via the cross-matrix method. In experiments, we evaluated our model on the typical task of link prediction. The results of our evaluation show that our approach outperforms state-of-the-art models.
