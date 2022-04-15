# Ontology Alignment Project Code

> This repository contains the code for a Graduate Coursework Project (EE8209 - Intelligent Systems) for Winter 2022.
## 1. Introduction and Setup
In this project we used two ontologies (mouse and human anatomy). We first need to preprocess the concept labels to create embeddings for concept labels. For this we used a pretrained model. The model should be downloaded from [http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin](http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin) and placed inside the `data` directory.
### 1.1. Needed Packages
You should install the following packages using pip (or any other python package manager of your choice)
- Pandas = 1.2.4
- Numpy = 1.19.5
- Matplotlib = 3.4.2
- PyTorch = 1.10.0
- pyg = 2.0.2
- pytorch-scatter = 2.0.9
- pytorch-sparse = 0.6.12
- scikit-learn = 1.0.1
- joblib = 1.1.0
- Graphviz = 2.49.3
- python-graphviz (from conda-forge) = 0.17
- fastdist = 1.1.3
- numba = 0.55.1
- gensim = 4.1.2
- tqdm = 4.64.0
- spacy = 3.2.4
- nltk = 3.7
- bs4 = 0.0.1

### 1.2. Files
The files are:

- `concept_preprocessing.ipynb`: The file contains the code for preprocessing the concepts and creating concept embeddings
- `training_autoencoder.ipynb`: The file contains the code to train a GNNAutoEncoder
- `create_representations.py`: The file contains the code to generate concept representations using the trained GNNAutoEncoder
- `calculate_cosine_similarity.py`: The file contains the code to calculate pairwise cosine similarity for concepts. The file calculates these similarities once for the embeddings and once for the representations
- `training_classifier.py`: The file contains the code to train a MNNClassifier and Naive Bayes Classifier once using the embeddings and once using the representations. It also uses the cosine similarities to calculate the performance of using cosine similarity for ontology alignment using embeddings and representations
- `models\autoencoder.py`: The file contains two GNNAutoEncoders we trained using PyTorch Geometric
- `models\classifier.py`: The file contains two MNNClassifiers we trained using PyTorch for ontology matching using embeddings and representations
- `utils\gnn_dataset.py`: The file contains the graph dataset created for each concept using PyTorch Geometric

## 2. Dataset
The dataset should be downloaded from [http://oaei.ontologymatching.org/2019/anatomy/anatomy-dataset.zip](http://oaei.ontologymatching.org/2019/anatomy/anatomy-dataset.zip). It contains 3 files. All three files should be put inside a folder named `anatomy` and the `anatomy`directory  should be placed inside the `data` directory.

The dataset is the ontologies of mouse and human anatomy. It also contains a reference file indicating the matched pairs of concepts between the two ontologies.

## 3. Running the Code
After installing the packages and placing the required files in the `data` directory, we need to first preprocess the concepts. To do so the files should be executed in the following order:

1. concept_preprocessing.ipynb
2. training_autoencoder.ipynb
3. create_representations.py
4. calculate_cosine_similarity.py
5. training_classifier.py

After running the files, all results will be presented and the figures will be generated and put in the `data` directory. The code can use `cuda` or `cpu` depending on the hardware availability.

## 4. Results
We first need to create graphs for each concept based on the concepts related to it. Figure 1 shows a sample concept graph with the blue node as the main concept and the blue nodes as the related concepts.

![Figure 1. Sample Concept Graph](/data/sample_graph_5088.png)

Figure 1. Sample Concept Graph

Then we need to decide which GNNAutoEncoder is better. To do so, we trained four different GNNAutoEncoders. These autoencoders are:

- Shallow 1: The model has one GNN encoder and one GNN decoder each having 128 hidden neurons
- Shallow 2: The model has one GNN encoder and one GNN decoder each having 64 hidden neurons
- Deep 1: The model has two GNN encoders, the first having 128 hidden neurons and the second having 64 hidden neurons, and two GNN decoders, the first having 64 hidden neurons and the second having 128 hidden neurons
- Deep 2: The model has two GNN encoders, the first having 64 hidden neurons and the second having 32 hidden neurons, and two GNN decoders, the first having 32 hidden neurons and the second having 64 hidden neurons

We used MSE loss to find the reconstruction loss of our GNNAutoEncoders. As can be seen in figure 2, Shallow 2 has the lowest MSE loss (0.0009691768992242628), thus was selected as the used GNNAutoEncoder.

![Figure 2. GNNAutoEncoder Models Comparison](/data/ae_final_validation_loss_comparison.png)

Figure 2. GNNAutoEncoder Models Comparison

The network structure of the GNN can be seen in figure 3.
![Figure 3. GNNAutoEncoder Network](/data/gnn_auto_encoder_network.png)

Figure 3. GNNAutoEncoder Network

We used this network to create representations for each concept embedding. Then we used MNN classifiers and Naive Bayes classifiers to classify a pair of concepts as aligned or not aligned. Figure 4 show the F1 Score comparison of these classifiers. Each classifier was once trained to use the GNN representations and once to only use the embeddings.

![Figure 4. Classifiers F1 Score Comparison](/data/classifiers_f1_score_Comparison.png)

Figure 4. Classifiers F1 Score Comparison

Based on the result, when we used the GNN representations to ontology matching using MNN Classifiers, we got the highest F1 Score.

Finally, we compared these models with using cosine similarity for ontology alignment. Figure 5 shows the F1 Score results. Based on the figure, using GNN representations with cosine similarity we achieved the highest F1 Score of 0.97.

![Figure 5. Comparison of Different Methods](/data/methods_f1_score_comparison.png)

Figure 5. Comparison of Different Methods

Table 1 shows the precision, recall, F1 score, and accuracy of all our methods.

Table 1. Methods Comparison

| Method                       | Precision | Recall | F1 Score | Accuracy |
|------------------------------|-----------|--------|----------|----------|
| MNN + Representations        | 0.87      | 0.87   | 0.87     | 0.87     |
| MNN + Embeddings             | 0.86      | 0.86   | 0.86     | 0.86     |
| NB + Representations         | 0.85      | 0.85   | 0.85     | 0.85     |
| NB + Embeddings              | 0.84      | 0.84   | 0.84     | 0.84     |
| Cosine Sim + Representations | 0.97      | 0.97   | 0.97     | 0.97     |
| Cosine Sim + Embeddings      | 0.96      | 0.96   | 0.96     | 0.96     |

## 5. Conclusion
Based on the results we provided above, when using the GNNAutoEncoder to create concept representations, we have seen an improvement in all methods. The GNNAutoEncoder uses the structural dependencies between each concept and its related concepts to create the representations. As a results, when matching concepts, the model can better decide if a pair of concepts are aligned or not.

## 6. Contributors
- [Hirad Daneshvar](https://github.com/hi-rad)
- [Hamed Karimi](https://github.com/hamedmx)
