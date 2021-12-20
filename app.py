import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import warnings

app = Flask(__name__)
app.secret_key = 'pPbLJZRi2JdnCoDkyA6gSyO4RePOFIPL'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'bestmodel.h5'

def load_model_with_compatibility():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        try:
            from tensorflow.keras.applications.mobilenet import MobileNet
            from tensorflow.keras.layers import Flatten, Dense
            from tensorflow.keras.models import Model
            
            base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
            
            for layer in base_model.layers:
                layer.trainable = False
                
            x = Flatten()(base_model.output)
            x = Dense(units=1, activation="sigmoid")(x)
            model = Model(base_model.input, x)

            try:
                model.load_weights(MODEL_PATH)
                return model, "Model architecture was recreated and weights were loaded successfully."
            except:
                return None, f"Could not load model weights. Original error: {str(e)}"
        except Exception as e2:
            return None, f"Failed to recreate model. Error: {str(e2)}. Original error: {str(e)}"

model, model_message = load_model_with_compatibility()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    if model is None:
        return {
            'error': True,
            'message': model_message
        }
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array)[0][0]
        
        return {
            'error': False,
            'probability': float(prediction),
            'has_tumor': bool(prediction >= 0.5),
            'class_name': 'Tumor' if prediction >= 0.5 else 'No Tumor',
            'confidence': float(prediction) if prediction >= 0.5 else float(1 - prediction)
        }
    except Exception as e:
        return {
            'error': True,
            'message': f"Error during prediction: {str(e)}"
        }

@app.route('/')
def home():
    if model is None:
        flash(f"Model could not be loaded: {model_message}")
    return render_template('index.html', model_error=(model is None))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash(f"Model could not be loaded: {model_message}")
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result = predict_image(file_path)
        
        if result.get('error', False):
            flash(result.get('message', 'An error occurred during prediction'))
            return redirect(url_for('home'))
        
        return render_template('result.html', 
                              filename=filename, 
                              probability=result['probability'],
                              has_tumor=result['has_tumor'],
                              class_name=result['class_name'],
                              confidence=result['confidence'] * 100)
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

@app.route('/about')
def about():
    return render_template('about.html', model_error=(model is None))

if __name__ == '__main__':
    if model is None:
        print(f"WARNING: Model could not be loaded: {model_message}")
        print("The application will run, but predictions will not be available.")
    app.run(debug=True)
