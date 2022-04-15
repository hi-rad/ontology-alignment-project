from typing import List
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
import json
import numpy as np


class ConceptGraphDataset(InMemoryDataset):

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'data')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'data', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = [
            'mouse.json', 'human.json'
        ]
        return names

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        data_list = ConceptGraphDataset.__create_data(osp.join(self.raw_dir, 'mouse.json'), ontology='mouse',
                                                      data_list=data_list)
        data_list = ConceptGraphDataset.__create_data(osp.join(self.raw_dir, 'human.json'), ontology='human',
                                                      data_list=data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def __create_data(file_path, ontology, data_list=None):
        if data_list is None:
            data_list = []
        with open(file_path, "r") as file:
            concepts = json.loads(file.read())
            for concept_id in concepts:
                concept = concepts[concept_id]
                if concept['alignable'] == 1 and len(concept['embed']) > 0:
                    x = []
                    mappings = []
                    edge_index = [[], []]
                    concept_emb = list(np.mean(np.array(concept['embed']), axis=0).astype(np.double))
                    x.append(concept_emb)
                    mappings.append(concept_id)
                    counter = 1
                    for related in concept['related']:
                        related_concept = concepts[related['code']]
                        if len(related_concept['embed']) > 0:
                            edge_index[0].append(0)
                            edge_index[1].append(counter)
                            edge_index[0].append(counter)
                            edge_index[1].append(0)
                            related_concept_emb = list(
                                np.mean(np.array(related_concept['embed']), axis=0).astype(np.double))
                            x.append(related_concept_emb)
                            mappings.append(related['code'])
                            counter += 1
                    x = np.array(x)
                    x = torch.from_numpy(x).float()
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                    data = Data(x=x,
                                edge_index=edge_index,
                                mappings=mappings,
                                ontology=ontology)
                    data_list.append(data)
        return data_list

    def __repr__(self) -> str:
        return 'concept_graph()'
