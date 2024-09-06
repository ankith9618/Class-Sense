from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import cv2
import face_recognition

app = Flask(__name__)
VIDEO_FOLDER = 'video'
FACES_FOLDER = 'static\\faces'
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER

# Ensure the folders exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

video_counter = 0
total_faces=0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/application')
def frontend():
    return render_template('frontend.html')

@app.route('/video', methods=['POST'])
def save_video():
    global video_counter
    global total_faces
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected video file"}), 400

    video_counter += 1
    video_filename = f"{video_counter}.mp4"
    video_path = os.path.join(app.config['VIDEO_FOLDER'], video_filename)
    video.save(video_path)
    
    total_faces = 0
    matched_faces = process_video(video_path)
    
    # Print count of persons found and their paths
    print(f"Total Faces Found : {total_faces}")
    print(f"Total Matches: {len(matched_faces)}")
    print("Matched faces:", matched_faces)
    
    # Remove the video file after processing
    os.remove(video_path)
    
    return jsonify({"message": "Video processed successfully", "matched_faces": matched_faces, "total_faces": total_faces}), 200

def load_known_faces(folder_path):
    """Load and encode faces from the given folder."""

    person_images = os.listdir(folder_path)
    face_encodings = []
    face_paths = []

    for image_path in person_images:
        image_full_path = os.path.join(folder_path, image_path)
        image = face_recognition.load_image_file(image_full_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])
            face_paths.append(image_full_path)
    
    return face_encodings, face_paths

def process_video(video_path):
    """Process the video to detect and match faces."""
    matched_faces = []
    video_capture = cv2.VideoCapture(video_path)
    frame_skip = 10  # Process every 10th frame
    frame_count = 0
    global total_faces
   
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # small_frame = resize_frame(frame)
            face_encodings = get_face_encodings(frame)
            total_faces = max(total_faces,len(face_encodings))
            matches = compare_faces(face_encodings)
            matched_faces.extend(matches)
        
        frame_count += 1
    
    video_capture.release()
    return list(set(matched_faces))  # Ensure unique paths

def resize_frame(frame, scale=0.25):
    """Resize the given frame to reduce processing time."""
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)

def get_face_encodings(frame):
    """Find face locations and encodings in the given frame."""
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    return face_encodings

def compare_faces(face_encodings):
    """Compare detected face encodings with known face encodings."""
    matched_faces = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
        for i, match in enumerate(matches):
            if match:
                matched_faces.append(known_face_paths[i])
                break
    return matched_faces

if __name__ == "__main__":
    # Load known faces at startup
    known_face_encodings, known_face_paths = load_known_faces(app.config['FACES_FOLDER'])
    app.run(debug=True)