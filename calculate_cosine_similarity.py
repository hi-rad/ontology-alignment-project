import pandas as pd

from utils import ConceptGraphDataset
import os
import joblib
from fastdist import fastdist
import numpy as np


def main():
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    dataset = ConceptGraphDataset(root=cwd)

    mouse = []
    human = []
    for i in range(0, len(dataset)):
        data = dataset[i]
        ontology = data.ontology
        if ontology == 'human':
            human.append(data)
        elif ontology == 'mouse':
            mouse.append(data)
        else:
            raise Exception('Ontology Not Supported!')

    print('Calculating Pairwise Cosine Similarity, Please Wait!')
    h_rep = []
    h_emb = []
    m_rep = []
    m_emb = []
    h_key = []
    m_key = []
    for i in range(0, len(human)):
        human_concept = human[i]
        h_concept_id = human_concept.mappings[0]
        h_key.append(h_concept_id)
        h_rep.append(list(joblib.load(pjoin(cwd, 'data', 'representations', 'human', f'{h_concept_id}.pkl'))))
        h_emb.append(human_concept.x[0].cpu().detach().numpy())

    for i in range(0, len(mouse)):
        mouse_concept = mouse[i]
        m_concept_id = mouse_concept.mappings[0]
        m_key.append(m_concept_id)
        m_rep.append(list(joblib.load(pjoin(cwd, 'data', 'representations', 'mouse', f'{m_concept_id}.pkl'))))
        m_emb.append(mouse_concept.x[0].cpu().detach().numpy())

    h_rep = np.array(h_rep).astype(np.double)
    m_rep = np.array(m_rep).astype(np.double)
    h_emb = np.array(h_emb).astype(np.double)
    m_emb = np.array(m_emb).astype(np.double)

    rep_sim = fastdist.matrix_to_matrix_distance(np.array(h_rep), np.array(m_rep), fastdist.cosine, "cosine")
    emb_sim = fastdist.matrix_to_matrix_distance(np.array(h_emb), np.array(m_emb), fastdist.cosine, "cosine")
    rep_sim_df = pd.DataFrame(list(rep_sim), columns=m_key, index=h_key)
    emb_sim_df = pd.DataFrame(list(emb_sim), columns=m_key, index=h_key)
    rep_sim_df.to_csv(pjoin(cwd, 'data', 'representations_similarity.csv'))
    emb_sim_df.to_csv(pjoin(cwd, 'data', 'embeddings_similarity.csv'))

    print('Pairwise Similarity Calculated Successfully!')


if __name__ == "__main__":
    main()
