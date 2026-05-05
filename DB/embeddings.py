import faiss
import numpy as np
import pickle
import os
class Embeddings:
	def __init__(self, index_path=None, emb_dim=512, labels={}, precision='float16'):
		self.emb_dim = emb_dim
		self.precision = precision
		self.labels = labels
		if index_path is not None:
			self.index = faiss.read_index(index_path)
		else:
			self.index = faiss.IndexFlatIP(emb_dim) 
	
	def load_emd(self, precision=32, efSearch=32):
		self.embed = np.load(self.storage_path).astype(precision)
		faiss.normalize_L2(embeddings)
		d = embeddings.shape[1]
		assert self.emb_dim == d
		self.index = faiss.IndexHNSWFlat(d, f'float{precision}')
		self.index.hnsw.efSearch = efSearch
		self.index.add(embeddings)
		
	def load_index(self, index_path):
		self.index = faiss.read_index(index_path)
		
	def add(self, new_embeddings):
		self.index.add(new_embeddings)
	
	def save_index(self, index_path):
		faiss.write_index(self.index, index_path)
		
	def save_index_from_embed(self, embed=None, index_path='/mnt/d/dataset/indexes/faces.index', embed_path=None):
		embed = faiss.normalize_L2(embed)
		self.index.add(embed)
		self.save_index(index_path)
		
	def load_labels(self, labels_path):
		with open(labels_path, "rb") as f:
			self.labels = pickle.load(f)
			
		
