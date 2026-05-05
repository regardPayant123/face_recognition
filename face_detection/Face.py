from insightface.app import FaceAnalysis
import cv2
import numpy as np

class Face:
	def __init__(self, model_name="buffalo_l", device="cpu", ):
		self.app = FaceAnalysis(name=model_name)  # modèle
		self.model_name=model_name
		if device == "cuda":
			self.app.prepare(ctx_id=0)  # 0 = GPU, -1 = CPU
		else:
			self.app.prepare(ctx_id=-1)
			
			
	def detect(self, frame):
		return self.app.get(frame) 
		
	def get_bbox(self, faces):
		bboxes = []
		for face in faces:
			bboxes.append((face.bbox,face.det_score))          #détection
		return bboxes
             
	def get_landmarks(self, faces):
		land_marks = []
		for face in faces:
			land_marks.append(face.kps)             # landmarks 5 points
		return land_marks
				
	def get_embed(self, faces):
		embedding = []
		for face in faces:
			embedding.append(face.embedding)    # vecteur 512D
		return np.array(embedding).astype("float32")	
			
	def draw(self, frame, detections, landmarks=None):
		"""
		Draw bounding boxes on frame
		"""
		for ((x1, y1, x2, y2), conf) in detections:
		    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
		    cv2.putText(frame, f"conf {conf:.2f}",
		                (int(x1), int(y2) + 10),
		              	cv2.FONT_HERSHEY_SIMPLEX,
		                0.5, (0,255,0), 2)
		    if landmarks != None:
		    	self.draw_landmarks(frame,landmarks)
		return frame
		
	def draw_landmarks(self, frame, landmarks):
		for landmark in landmarks:
			for (x,y) in landmark:
				cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
		return frame
				
	def draw_name(self,frame, name, best_sim, detections):
		# draw the person name
		for i, ((x1, y1, x2, y2), conf) in enumerate(detections):
			#print(f"{name[i]} {best_sim[i]:.2f} {x1+10}")
			cv2.putText(frame, f"{name[i]} {best_sim[i]:.2f}%similar", 
				(int(x1+10), int(x2-10)),
				cv2.FONT_HERSHEY_SIMPLEX, 
				0.6, (0,255,0), 2)

		return frame	
		
	def detect_and_draw(self, frame, with_landmarks = True, with_name=False):
		"""
		Convenience method
		"""
		faces = self.detect(frame)
		detections = self.get_bbox(faces)
		landmarks = self.get_landmarks(faces)
		
		if with_landmarks and with_name:
			return self.draw(frame, detections, landmarks)
		else:
			return self.draw(frame, detections)
	


