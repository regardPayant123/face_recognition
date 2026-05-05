import cv2
import threading
import queue
from DB import *
from face_detection import *
import faiss



frame_queue = queue.Queue(maxsize=1)
index_path = '/mnt/d/dataset/indexes/faces.index'
label_path = '/mnt/d/dataset/indexes/labels.pkl'
name = []
name_ids = []
sim_score = []
print("starting face detection")

#  detection_model = YOLOv8Face(model_path="/home/enama/Downloads/yolov8n-face.pt")
detection_model = Face()
database = Embeddings()
database.load_index(index_path)
database.load_labels(label_path)
k = 5

# thread capture
threading.Thread(
    target=capture_frame_in_queue,
    args=(frame_queue,),
    daemon=True
).start()

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()


        # inference
        detections = detection_model.detect(frame)
        bbox = detection_model.get_bbox(detections)
        landmark = detection_model.get_landmarks(detections)
        if len(bbox) != 0:
           embeddings = detection_model.get_embed(detections).reshape(-1, database.emb_dim)
           faiss.normalize_L2(embeddings)
           dists, ids = database.index.search(embeddings, k)
	#TODO: complete the function for research in database
           frame = detection_model.draw(frame, bbox, landmark)
           for i, j in zip(dists,ids):
               sim_score.append(i[0]) 
               name_ids.append(j[0])
               if sim_score[-1] > 0.5:
                   name.append(database.labels[name_ids[-1]])
               else:
                   name.append("unknown")	
           frame = detection_model.draw_name(frame, name, sim_score, bbox)	
		
        cv2.imshow("WSL Stream", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
