from utils import ConceptGraphDataset
import torch
import os
import graphviz
from random import randint
from fastdist import fastdist
import numpy as np


def main():

    u = np.random.rand(100)
    v = np.random.rand(100)

    print(fastdist.euclidean(u, v))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # input1 = torch.randn(1, 64).to(device)
    # input2 = torch.randn(1, 64).to(device)
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # cos = cos.to(device)
    # output = cos(input1, input2)
    # print(output.size())
    # print(output)
    # output = cos(input2, input1)
    # print(output.size())
    # print(output.cpu().detach().numpy())
    # cwd = os.getcwd()
    # path = os.path
    # pjoin = path.join
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dataset = ConceptGraphDataset(root=cwd)
    # print(dataset)
    #
    # graph_index = randint(0, len(dataset))
    # sample_graph = dataset[graph_index]
    # print(sample_graph)
    #
    # number_of_nodes = sample_graph.num_nodes
    # number_of_edges = sample_graph.num_edges
    #
    # dot = graphviz.Graph(comment=f'Dataset: {dataset}')
    #
    # nodes = []
    # for node in range(number_of_nodes):
    #     node_name = sample_graph.mappings[int(node)]
    #     nodes.append(str(node_name))
    #
    # with dot.subgraph() as subgraph:
    #     subgraph.attr(rank='same')
    #     label = nodes[0]
    #     subgraph.node_attr.update(style='filled', fontcolor='white', fillcolor='#4287f5')
    #     subgraph.node('0', label)
    #
    # with dot.subgraph() as subgraph:
    #     subgraph.attr(rank='same')
    #     for node in range(1, len(nodes)):
    #         label = nodes[node]
    #         subgraph.node_attr.update(style='filled', fontcolor='white', fillcolor='#0e9c47')
    #         subgraph.node(f'{node}', label)
    #
    # edges = []
    # for edge in range(number_of_edges):
    #     first_node = min(int(sample_graph.edge_index[0][edge]), int(sample_graph.edge_index[1][edge]))
    #     second_node = max(int(sample_graph.edge_index[0][edge]), int(sample_graph.edge_index[1][edge]))
    #     if f'{first_node}_{second_node}' not in edges:
    #         edges.append(f'{first_node}_{second_node}')
    #
    # for edge_item in edges:
    #     first_node, second_node = edge_item.split('_')
    #     dot.edge(str(first_node), str(second_node))
    #
    # dot.render(
    #     pjoin(cwd, 'data', f'sample_graph_{graph_index}'),
    #     view=True,
    #     cleanup=True,
    #     format='png'
    # )


if __name__ == "__main__":
    main()
