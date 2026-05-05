from DB import *
from face_detection import *
import faiss
import cv2
import pickle
import numpy as np
import os
from tqdm import tqdm

img_dir = "/mnt/d/dataset/labeled_faces_in_the_wild/lfw-funneled/lfw_funneled"
labels_path = '/mnt/d/dataset/indexes/labels.pkl'
embed_path = '/mnt/d/dataset/indexes/embeddings.npy'
index_dir = '/mnt/d/dataset/indexes/faces.index'
emb_index = Embeddings()
model = RetinaFace()
embeddings = []
labels = []


directories = [
    os.path.join(img_dir, nom) for nom in os.listdir(img_dir)
    if os.path.isdir(os.path.join(img_dir, nom))
]
# print("here is the name:",directories[0])
for directory in tqdm(directories):
	person = os.path.basename(directory)

	for img_f in os.listdir(directory):
		if img_f.lower().endswith((".jpg", ".jpeg")):
			path = os.path.join(directory, img_f)

			img = cv2.imread(path)
			if img is None:
				continue
			detection = model.detect(img)

			if len(detection) == 1:
				embeddings.append(detection[0].embedding)
				labels.append(person)
				# print("here is the name:",person)
			
embeddings = np.array(embeddings).astype("float32")
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

ids = np.arange(len(embeddings))

emb_index.index = faiss.IndexIDMap(emb_index.index)
emb_index.index.add_with_ids(embeddings, ids)

np.save(embed_path, embeddings)

label_dict = {i: labels[i] for i in range(len(labels))}
with open(labels_path, "wb") as f:
    pickle.dump(label_dict, f)

emb_index.save_index(index_dir)

print(f"{len(embeddings)} samples identified and isolated")
print("index creation over")
