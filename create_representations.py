from utils import ConceptGraphDataset
import torch
from models import ShallowGNNAutoEncoder
import os
from torch_geometric.loader import DataLoader
import joblib


def main():
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ConceptGraphDataset(root=cwd)

    model = ShallowGNNAutoEncoder(in_channels=dataset.num_features, bottleneck_channels=64)
    model.load_state_dict(torch.load(pjoin(cwd, 'data', 'shallow_ae_64.pt')))
    model.to(device)
    model.eval()

    if not os.path.exists(pjoin(cwd, 'data', 'representations')):
        os.mkdir(pjoin(cwd, 'data', 'representations'))
    print('Creating Representations, Please Wait!')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        data = data.to(device)
        out = model.encoder(data.x, data.edge_index)
        ontology = data.ontology[0]
        concept_id = data.mappings[0][0]
        representation = out.cpu().detach().numpy()[0]
        if not os.path.exists(pjoin(cwd, 'data', 'representations', ontology)):
            os.mkdir(pjoin(cwd, 'data', 'representations', ontology))
        joblib.dump(representation, pjoin(cwd, 'data', 'representations', ontology, f'{concept_id}.pkl'))

    print('Representations Created Successfully!')


if __name__ == "__main__":
    main()
