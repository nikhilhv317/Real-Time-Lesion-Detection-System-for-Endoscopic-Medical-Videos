import os
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time
import subprocess

# Setup logging
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 100 * 1024 * 1024

# Global model and device
MODEL = None
DEVICE = None
MODEL_PATH = r"E:/ENDO_Project/lesion_detector.pth"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        return int(result.stdout.strip().replace('%', ''))
    except Exception as e:
        logging.warning(f"Failed to get GPU utilization: {e}")
        return None

def load_model(model_path, num_classes=2):
    if not torch.cuda.is_available():
        logging.error("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        logging.info(f"Model on {device}, Memory: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB, Utilization: {get_gpu_utilization()}%")
    return model, device

def initialize_model():
    global MODEL, DEVICE
    try:
        MODEL, DEVICE = load_model(MODEL_PATH)
        logging.info("Model initialized successfully")
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        raise

def draw_boxes(frame, boxes, labels, scores, confidence_threshold=0.5):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold and label == 1:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
            draw.text((box[0], box[1] - 20), f"lesion: {score:.2f}", fill="red", font=font)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame, model, transform, device, confidence_threshold=0.5):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            predictions = model(img_tensor)[0]
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
        frame = draw_boxes(frame, boxes, labels, scores, confidence_threshold)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            logging.debug(f"Frame processed on {device}, Memory: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB, Utilization: {get_gpu_utilization()}%")
        return frame
    except Exception as e:
        logging.error(f"Frame processing failed: {e}")
        raise

def generate_frames(video_path, model, transform, device, start_time=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 25) > 0.1:
        logging.warning(f"Video FPS is {fps:.2f}, expected 25 FPS. Playback may be inconsistent.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    frame_count = 0
    last_frame_time = time.time()
    target_frame_time = 1.0 / 25  # 40ms for 25 FPS
    logging.info(f"Streaming: {video_path}, start: {start_time}s, target FPS: 25")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info(f"Stream ended: {video_path}, frames: {frame_count}")
                break
            frame_count += 1
            start_process = time.time()
            try:
                processed_frame = process_frame(frame, model, transform, device)
                ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    logging.error(f"Failed to encode frame {frame_count}")
                    continue
                process_time = time.time() - start_process
                current_time = time.time()
                actual_fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                last_frame_time = current_time
                logging.debug(f"Frame {frame_count} processed in {process_time:.3f}s, FPS: {actual_fps:.2f}")
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                sleep_time = max(0, target_frame_time - process_time)
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
                continue
    finally:
        cap.release()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logging.info(f"VideoCapture released for: {video_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            logging.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov'}), 400
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_FILE_SIZE:
            logging.error(f"File too large: {file_size} bytes")
            return jsonify({'error': 'File too large (max 100MB)'}), 400
        if file_size == 0:
            logging.error("Empty file uploaded")
            return jsonify({'error': 'Empty file uploaded'}), 400
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(video_path)
            logging.info(f"File saved: {video_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        return jsonify({'filename': filename}), 200
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return jsonify({'error': 'Video not found'}), 404
    logging.info(f"Serving video: {video_path}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/stream/<string:filename>')
def stream(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return jsonify({'error': 'File not found'}), 404
    start_time = float(request.args.get('t', '0'))
    if MODEL is None or DEVICE is None:
        logging.error("Model not initialized")
        return jsonify({'error': 'Model not initialized'}), 500
    transform = transforms.Compose([transforms.ToTensor()])
    logging.info(f"Starting stream for: {video_path}, time: {start_time}s")
    return Response(generate_frames(video_path, MODEL, transform, DEVICE, start_time), mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize model at startup
try:
    initialize_model()
except Exception as e:
    logging.critical(f"Server startup failed: {e}")
    exit(1)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
