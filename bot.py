import numpy as np
import pandas as pd
import faiss
import torch
# from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import os
import glob
# from torch.utils.data import Dataset
# from json import JSONEncoder
# import json
# from collections import defaultdict

torch_device = 'cuda' if torch.cuda.is_available() else "cpu"
if not os.path.exists('corpus_embeddings.csv'):

    from sentence_transformers import SentenceTransformer

    def split_text(text):
        return text.split('')[-1]
    all_files = glob.glob('*.csv')
    filename = []
    for files in all_files:
        column_names = ['Question', 'Answer']
        df = pd.read_csv(files, names=column_names)
        df.dropna(inplace=True)
        df['Answer'] = df['Answer'].map(lambda text: split_text(text))
        filename.append(df)
    corpus = pd.concat(filename, ignore_index=True)
    corpus.reset_index(drop=True, inplace=True)
    encoder = SentenceTransformer('bert-base-nli-mean-tokens', device=torch_device)
    corpus_embeddings = encoder.encode(corpus['Question'].tolist())
    np.savetxt('corpus_embeddings.csv', corpus_embeddings, delimiter=',')

    # for index, question in enumerate(corpus['Question'].tolist()):
    #     weights_dict[question] = corpus_embeddings[index]
    # with open("corpus_embeddings.json", "w") as outfile:
    #     json.dump(weights_dict, outfile, indent=4, sort_keys=False, cls=NpEncoder)

else:
    corpus_embeddings = np.loadtxt('corpus_embeddings.csv', delimiter=',')
    
def knowledgeBase_cosine():
    # input_prompt = 'Have you ever had a dream?'
    input_prompt = input('Enter question: ')
    text_embedding = encoder.encode([input_prompt], device=torch_device)
    similarities = 1 - cdist(text_embedding, corpus_embeddings, 'cosine')
    similarities = np.around(similarities, decimals=2)
    best_sim_idx = np.argmax(similarities[0])
    most_similar_question = corpus.loc[best_sim_idx].Question
    answer = corpus.loc[best_sim_idx].Answer
    similarity_value = similarities[0].max()
    print(f'Prompt: {input_prompt}, Closest question: {most_similar_question}, Answer: {answer}, Sim value: {similarity_value}')

def knowledgeBase_faiss():

    embedding_dim = corpus_embeddings[0].size
    corpus_embeddings_copy = corpus_embeddings.copy().astype(np.float32)
    corpus_ids = np.arange(0, len(corpus_embeddings)).astype(np.int64)
    faiss.normalize_L2(corpus_embeddings_copy)
    index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(corpus_embeddings_copy, corpus_ids)
    '''
    Implementing Faiss search using GPU
    '''
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    k_nearest = 1
    input_prompt = " "
    while True:
        # input_prompt = 'Have you ever had a dream?'
        input_prompt = input('Enter question: ')
        if input_prompt == "exit()":
            break
        text_embedding = encoder.encode([input_prompt], device=torch_device).astype(np.float32)
        faiss.normalize_L2(text_embedding)
        similarities, similarities_ids = gpu_index.search(text_embedding, k = k_nearest)
        similarities = np.around(np.clip(similarities[0][0], 0, 1), decimals=4)
        best_sim_idx = similarities_ids[0][0]
        most_similar_question = corpus.loc[best_sim_idx].Question
        answer = corpus.loc[best_sim_idx].Answer
        print(f'Prompt: {input_prompt}, Closest question: {most_similar_question}, Answer: {answer}, Sim value: {similarities}')


        
if __name__ == "__main__":
    knowledgeBase_faiss()