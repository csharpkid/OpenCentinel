import threading
import platform
from flask import Flask, Response, render_template, request, jsonify
import time
import requests
from collections import deque
import base64
from openai import OpenAI  # Added for OpenAI integration
import os  # Added for environment variable handling
import json
import cv2
import numpy as np

# Set OpenAI API key (preferably from environment variable)
openai_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)

class MotionDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.buffer_size = 4
        self.source = 0
        self.webhook_url = None
        self.last_notification_time = 0
        self.notification_cooldown = 5
        self.notifications = deque(maxlen=10)
        
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.initialize_camera()

    def open_camera_with_timeout(self, source, timeout=5):
        start_time = time.time()
        if isinstance(source, int):
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
            cap = cv2.VideoCapture(source, backend)
        else:
            cap = cv2.VideoCapture(source)
        
        while not cap.isOpened() and (time.time() - start_time) < timeout:
            cap.release()
            if isinstance(source, int):
                cap = cv2.VideoCapture(source, backend)
            else:
                cap = cv2.VideoCapture(source)
            time.sleep(0.1)
            
        if not cap.isOpened():
            print(f"Timeout al abrir fuente {source}")
            return None
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return cap

    def is_frame_black(self, frame):
        return np.mean(frame) < 10

    def initialize_camera(self):
        print("Inicializando cámara...")
        self.cap = self.open_camera_with_timeout(self.source)
        if not self.cap or not self.cap.isOpened():
            raise ValueError("No se pudo abrir la cámara")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        print("Cámara inicializada correctamente")

    def send_webhook_notification(self, motion_count, face_count, motion_rois, face_rois, frame):
        if not self.webhook_url or (time.time() - self.last_notification_time) < self.notification_cooldown:
            return

        try:
            payload = {
                'event': 'motion_and_faces_detected',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'motion_objects': motion_count,
                'faces_detected': face_count,
                'motion_rois': [base64.b64encode(roi).decode('utf-8') for roi in motion_rois],
                'face_rois': [base64.b64encode(roi).decode('utf-8') for roi in face_rois],
                'frame': base64.b64encode(frame).decode('utf-8')
            }
            requests.post(self.webhook_url, json=payload, timeout=2)
            self.last_notification_time = time.time()
            print(f"Notificación enviada al webhook: {self.webhook_url}")
        except Exception as e:
            print(f"Error enviando notificación al webhook: {e}")

    def add_notification(self, motion_count, face_count, motion_rois, face_rois, frame):
        motion_rois_base64 = [base64.b64encode(cv2.imencode('.jpg', roi)[1].tobytes()).decode('utf-8') for roi in motion_rois]
        face_rois_base64 = [base64.b64encode(cv2.imencode('.jpg', roi)[1].tobytes()).decode('utf-8') for roi in face_rois]
        frame_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1].tobytes()).decode('utf-8')
        
        notification = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'motion_objects': motion_count,
            'faces_detected': face_count,
            'motion_rois': motion_rois_base64,
            'face_rois': face_rois_base64,
            'frame': frame_base64
        }
        self.notifications.appendleft(notification)
        self.send_webhook_notification(motion_count, face_count,
                                    [cv2.imencode('.jpg', roi)[1].tobytes() for roi in motion_rois],
                                    [cv2.imencode('.jpg', roi)[1].tobytes() for roi in face_rois],
                                    cv2.imencode('.jpg', frame)[1].tobytes())

    def get_notifications(self):
        return list(self.notifications)

    def generate_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or self.is_frame_black(frame):
                print("Error leyendo frame")
                continue

            frame = cv2.resize(frame, (640, 480))
            original_frame = frame.copy()
            
            # Detección de movimiento
            fg_mask = self.background_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
            fg_mask = cv2.erode(fg_mask, None, iterations=2)
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_count = 0
            motion_rois = []

            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    motion_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    margin = 10
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(frame.shape[1], x + w + margin)
                    y_end = min(frame.shape[0], y + h + margin)
                    motion_roi = original_frame[y_start:y_end, x_start:x_end]
                    motion_rois.append(motion_roi)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (165, 77, 17), 2)

            # Detección de rostros solo si hay movimiento, pero sin dibujar en la vista previa
            face_count = 0
            face_rois = []
            if motion_count > 0:
                gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    face_count += 1
                    margin = 10
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(frame.shape[1], x + w + margin)
                    y_end = min(frame.shape[0], y + h + margin)
                    face_roi = original_frame[y_start:y_end, x_start:x_end]
                    face_rois.append(face_roi)
                    # No dibujamos los rectángulos de rostros en la vista previa

            if motion_count > 0:
                self.add_notification(motion_count, face_count, motion_rois, face_rois, original_frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def set_webhook(self, url):
        self.webhook_url = url
        print(f"Webhook configurado: {url}")

    def close(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def analyze_image_with_openai(self, frame):
        """Analyze an image using OpenAI's gpt-4o-mini vision model."""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Send to OpenAI API with strict JSON formatting
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image and provide a detailed classification in strict JSON format:\n"
                                    "1. Detect the situation in Spanish (e.g., 'armas de fuego detectadas', 'actividad sospechosa', 'escena normal').\n"
                                    "2. Classify the severity level as 'Neutral', 'Medium', or 'Critical'.\n"
                                    "3. Describe the scene in Spanish, including emotions of people (if present) and their clothing.\n"
                                    "4. If vehicles are present, identify the model, color, and OCR any visible license plates.\n"
                                    "Return the response as a JSON object with keys: 'situation', 'severity', 'description', 'vehicles'. "
                                    "Ensure the response is a valid JSON string, enclosed in ```json``` marks, with no additional text outside the JSON."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300  # Increased to allow for detailed JSON response
            )
            # Extract and parse the JSON response
            result = response.choices[0].message.content.strip()
            # Remove ```json and ``` marks if present
            if result.startswith("```json") and result.endswith("```"):
                result = result[7:-3].strip()
            return json.loads(result)  # Parse the JSON response from OpenAI
        except Exception as e:
            print(f"Error analyzing image with OpenAI: {e}")
            return {
                "situation": "Error durante el análisis",
                "severity": "Neutral",
                "description": f"Error: {str(e)}",
                "vehicles": []
            }

detector = MotionDetector()

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_webhook', methods=['POST'])
def set_webhook():
    data = request.get_json()
    if not data or 'webhook_url' not in data:
        return jsonify({'error': 'URL de webhook requerida'}), 400
    
    webhook_url = data['webhook_url']
    detector.set_webhook(webhook_url)
    return jsonify({'message': 'Webhook configurado correctamente'}), 200

@app.route('/get_notifications', methods=['GET'])
def get_notifications():
    return jsonify(detector.get_notifications())

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze a single image."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Image data required'}), 400
    
    # Decode base64 image
    try:
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze with OpenAI
        analysis_result = detector.analyze_image_with_openai(frame)
        
        return jsonify(analysis_result), 200
    except Exception as e:
        return jsonify({
            "situation": "Error durante el procesamiento",
            "severity": "Neutral",
            "description": f"Error: {str(e)}",
            "vehicles": []
        }), 500

def run_app():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    try:
        server_thread = threading.Thread(target=run_app)
        server_thread.start()
        print("Servidor Flask iniciado en http://localhost:5000")
        
        while detector.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Cerrando aplicación...")
        detector.close()
        server_thread.join()