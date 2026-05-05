import socket
import struct
import pickle
import cv2
import queue
import threading
import time

frame_queue = queue.Queue(maxsize=1)

def capture_offline():
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if frame_queue.full():
		    frame_queue.get()

		frame_queue.put(frame)
		time.sleep(0.03)
		

def capture_online(HOST="172.20.192.1", PORT=9999):
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# IP Windows depuis WSL (souvent la gateway)
	client.connect((HOST, PORT))

	data = b""
	payload_size = struct.calcsize("Q")

	while True:
		while len(data) < payload_size:
			data += client.recv(4096)
		packed_size = data[:payload_size]
		data = data[payload_size:]

		msg_size = struct.unpack("Q", packed_size)[0]

		while len(data) < msg_size:
			data += client.recv(4096)

		frame_data = data[:msg_size]
		data = data[msg_size:]

		frame = pickle.loads(frame_data)
		
		cv2.imshow("WSL Stream", frame)
		if cv2.waitKey(1) == 27:
			break
	return frame
	
	
def capture_frame_in_queue(frame_queue, HOST="172.20.192.1", PORT=9999):

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))

    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            data += client.recv(4096)

        packed_size = data[:payload_size]
        data = data[payload_size:]

        msg_size = struct.unpack("Q", packed_size)[0]

        while len(data) < msg_size:
            data += client.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data)

        # garder seulement la dernière frame (latence minimale)
        if frame_queue.full():
            frame_queue.get()

        frame_queue.put(frame)
	
def capture_img(img_path, resize = None):
	img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return RGB_img

def capture_DB(sever_name, ID, pass_wrd):
	return NotImplementedError
	
