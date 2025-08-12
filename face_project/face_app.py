import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

##configuration

KNOWN_FACES_DIR= "known_faces"
SAVED_FACES_DIR= "saved_faces"
HAAR_SCALE_FACTOR= 1.1
HAAR_MIN_NEIGHBORS= 5
RECOGNITION_TOLERANCE = 0.48
FRAME_RESIZE_SCALE= 0.5

USE_DNN_DETECTOR = False
DNN_PHOTO_PATH ="deploy.prototxt"
DNN_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.5

#prepre directories

os.makedirs(SAVED_FACES_DIR, exist_ok=True)

#load

#HAAR cascade
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#dnn detecttor

dnn_net = None
if USE_DNN_DETECTOR:
    if not (os.path.exists(DNN_PHOTO_PATH) and os.path.exists(DNN_MODEL_PATH)):
        raise ValueError(
            f"DNN model files not found. place '{DNN_PHOTO_PATH}' and '{DNN_MODEL_PATH}' in the project folder or set USE_DNN_DETECTOR = False."
        )
    dnn_net = cv2.dnn.readNetFromCaffe(DNN_PHOTO_PATH, DNN_MODEL_PATH)

known_encodings =[]
known_names=[]

if not os.path.exists(KNOWN_FACES_DIR):
    raise ValueError(f"Known faces directory '{KNOWN_FACES_DIR}' not found. Create it and add images.")

for fname in os.listdir(KNOWN_FACES_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path=os.path.join(KNOWN_FACES_DIR, fname)
    image = face_recognition.load_image_file(path)
    encs=face_recognition.face_encodings(image)
    if len(encs)== 0:
        print(f"[WARN] no faces found in {fname}; skipping.")
        continue
    encoding = encs[0]
    label = os.path.splitext(fname)[0]
    known_encodings.append(encoding)
    known_names.append(label)
print(f"[INFO]Loaded {len(known_encodings)} known faces.")

#hlprs

def save_recognized_face(face_img_bgr, name):
    """Save the crooped face with timestamp for logging."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{name}_{ts}.jpg"
    path = os.path.join(SAVED_FACES_DIR, filename)
    cv2.imwrite(path, face_img_bgr)
    print(f"[INFO] saved recognized face to {path}")

def detect_faces_haar(frame_bgr):
    """"Return list of boxes (x, y, w, h) using Haar cascade. """
    gray = cv2.cvtColor(frame_bgr , cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray , scaleFactor = HAAR_SCALE_FACTOR, minNeighbors= HAAR_MIN_NEIGHBORS)
    return faces

def detect_faces_dnn(frame_bgr):
    """
    Return list of boxes (x, y, w, h) using DNN face detector.
    Note: frame_bgr is expected at original size.
     """
    (h, w) = frame_bgr.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    boxes =[]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DNN_CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7]* np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x = max(0, x1)
            y = max(0, y1)
            boxes.append((x, y, x2-x, y2-y))
    return boxes

def recognize_face(cropped_face_bgr):
    """
    Given a cropped face (BGR), compute embedding and compare to known faces.
    Returns (name, best_distance). Name is 'Unknown' if none within tolerance.
    """
    rgb = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if len(encs) == 0:
        return "Unknown" ,None
    enc = encs[0]
    distances = face_recognition.face_distance(known_encodings, enc) if len(known_encodings)> 0 else[]
    if len(distances) == 0:
        return "Unknown" , None
    best_idx = np.argmin(distances)
    best_dist = float(distances[best_idx])
    if best_dist <= RECOGNITION_TOLERANCE:
        return known_names[best_idx], best_dist
    return "Unknown", best_dist

#frame processing

def process_stream(source=0, is_image= False):
    """"
     source: webcam index (0) or video path or image path
    is_image: True when source is image path
    """
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print("[ERROR] Could not load image:", source)
            return
        frames= [frame]
        single_frame_mode = True
    else:
        single_frame_mode= False
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR]Could not open video source:", source)
            return
        
    while True:
        if single_frame_mode:
            frame = frames[0].copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        original_h, original_w = frame.shape[:2]
        if FRAME_RESIZE_SCALE != 1.0:
            small = cv2.resize(frame, (int(original_w * FRAME_RESIZE_SCALE), int(original_h * FRAME_RESIZE_SCALE)))
        else:
            small = frame.copy()

        if USE_DNN_DETECTOR and dnn_net is not None:
            boxes = detect_faces_dnn(frame)
            draw_scale = 1.0
        else:
            faces = detect_faces_haar(small)
            if FRAME_RESIZE_SCALE != 1.0:
                scale = 1.0 / FRAME_RESIZE_SCALE
                boxes = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h)in faces]
            else:
                boxes = faces

            
            for (x, y, w, h) in boxes:
                x1 = max(0, x); y1 = max(0, y); x2 = min(frame.shape[1], x + w); y2 = min(frame.shape[0], y + h)
                if x2 <= x1 or y2 <=y1:
                    continue
                cropped = frame[y1:y2, x1:x2]
                name, dist = recognize_face(cropped)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name}" if dist is None else f"{name} ({dist:.2f})"
                cv2.putText(frame, label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                if name != "Unknown":
                    save_recognized_face(cropped, name)

        cv2.imshow("Face Detection & Recognition", frame)
            
        if single_frame_mode:
            cv2.waitKey(0)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not single_frame_mode and 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()


##main menu

def main():
    print("Face Detection & Recognition")
    print("options:")
    print(" 1) webcam")
    print(" 2) video file")
    print(" 3) Image file")
    choice = input("choose mode(1/2/3):").strip()
    if choice =="1":
        print("[INFO] Starting webcam. press 'q' to quit.")
        process_stream(0, is_image=False)
    elif choice == "2":
        path = input("enter video file path:").strip()
        process_stream(path, is_image=False)
    elif choice == "3":
        path = input("enter image file path:").strip()
        process_stream(path, is_image=True)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()



