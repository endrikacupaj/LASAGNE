import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from dataset import CSQADataset
from graph import TypeRelationGraph

# import constants
from constants import *

class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.conv1 = GATConv(300, 300, heads=2, concat=False, dropout=0.6)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(300, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.relu(x)
        x = self.linear(x)
        return x

# data and model
dataset = CSQADataset()
graph = dataset.get_graph()
target = TypeRelationGraph(dataset.graph_field.vocab).type_mask
model = GraphModel().to(DEVICE)

# criterion and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# train model
model.train()
for i in range(300):
    optimizer.zero_grad()
    output = model(graph)
    loss = criterion(output.squeeze(), target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {i} ---- loss: {loss.item()}')
