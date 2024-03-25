from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2
import base64
import shutil
from flask import send_file
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'temp_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_modifications(image, crop_box=None, blur_radius=None, rotation=None, grayscale=False, edge_detection=False,
                        brightness=None, contrast=None):
    if crop_box:
        x1, y1, x2, y2 = crop_box
        image = image[y1:y2, x1:x2]
    if blur_radius is not None and blur_radius > 0:
        image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    if rotation:
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if edge_detection:
        image = cv2.Canny(image, 100, 200)  # Adjust parameters as needed
    if brightness is not None:
        image = cv2.convertScaleAbs(image, alpha=brightness)
    if contrast is not None:
        image = np.clip(image * contrast, 0, 255).astype(np.uint8)
    return image

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = cv2.imread(filename)
        
        crop_box = None
        blur_radius = None
        rotation = None
        grayscale = False
        edge_detection = False
        brightness = None
        contrast = None

        if 'crop' in request.form and request.form['crop']:
            crop_data = request.form['crop'].split(',')
            crop_box = tuple(map(int, map(round, map(float, crop_data))))
        if 'blur' in request.form and request.form['blur']:
            blur_radius = int(request.form['blur'])
        if 'rotate' in request.form and request.form['rotate']:
            rotation = int(request.form['rotate'])
        if 'grayscale' in request.form:
            grayscale = True
        if 'edge_detection' in request.form:
            edge_detection = True
        if 'brightness' in request.form:
            brightness = float(request.form['brightness'])
        if 'contrast' in request.form:
            contrast = float(request.form['contrast'])
        
        modified_img = apply_modifications(img, crop_box, blur_radius, rotation, grayscale, edge_detection,
                                           brightness, contrast)
        modified_img_b64 = encode_image(modified_img)

        return jsonify({'modified_img': modified_img_b64})

    return render_template('index.html')


@app.route('/clear_images', methods=['POST'])
def clear_images():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
        message = 'Images folder cleared successfully'
    except Exception as e:
        message = f'Error clearing images folder: {str(e)}'
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
