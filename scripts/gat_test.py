# %%
from csqa_dataset import CSQADataset

import torch
import torch.nn as nn
import torch.optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torchtext.data import BucketIterator

from ..knowledge_graph.knowledge_graph import KnowledgeGraph
from importlib import reload
import graph as graph_to_reload
reload(graph_to_reload)
from graph import Graph
# %%
kg = KnowledgeGraph("/data/joanplepi/csqa/data/kb/")

dataset = CSQADataset('/data/sample3')
# %%
type_vocab = dataset.get_vocabs()[TYPE]
pred_vocab = dataset.get_vocabs()[PREDICATE]

graph = Graph(type_vocab, pred_vocab)
graph.extract_graph_mode_full(kb)
print(len(type_vocab), len(pred_vocab))

# %%
edge_index = torch.tensor([graph.start, graph.end], dtype=torch.long, requires_grad=False)
print(len(graph.representations[0]))
data = Data(x=torch.cat(graph.representations), edge_index=edge_index)
print()
temp_data = Data(x=torch.cat(graph.representations), edge_index=edge_index)

class GraphModel(nn.Module):
    def __init__(self, data):
        super(GraphModel, self).__init__()
        self.data = data
        self.gat = GATConv(100, 100, heads=2, concat=False, dropout=0.2)
        self.linear_layer = nn.Linear(100, 1)

    def forward(self):
        out = self.gat(self.data.x, self.data.edge_index)
        return self.linear_layer(out)

graph_model = GraphModel(data.to(DEVICE)).to(DEVICE)
# %%
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(graph_model.parameters())
type_target = torch.FloatTensor(graph.type_mask).to(DEVICE)

for i in range(100):
    optimizer.zero_grad()
    output = graph_model()
    loss = criterion(output.squeeze(), type_target)
    print(f'Epoch {i} ---- loss: {loss.item()}')
    loss.backward(retain_graph=True)
    optimizer.step()