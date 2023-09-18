import os
from flask import Flask, jsonify, render_template, request
from flask import Markup
from werkzeug.utils import secure_filename
import tensorflow.keras as keras
import input_process
import predict_genre


app = Flask(__name__)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3','wav'}

#===============================================
import json

# Load the model architecture from config.json
with open('rnn.keras/config.json', 'r') as json_file:
    model_config = json_file.read()

model = keras.models.model_from_json(model_config)

# Load the model weights
model.load_weights('rnn.keras/model.weights.h5')
#==================================================


# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Main route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'songFile' not in request.files:
        jsonify({'message':'failure'})

    file = request.files['songFile']

    if file.filename == '':
        jsonify({'message':'failure'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        X_test=input_process.process(file_path)
        prediction=model.predict(X_test)
        predicted_genre=predict_genre.index(prediction)
        
        print("Predicted genre is: ",predicted_genre)
        
        upload_return={'message':f" the predicted genre is {predicted_genre}"}
        return jsonify(upload_return)

    return jsonify({'message':'failure'})


# last_selected_file=''


# @app.route('/upload_api', methods=['GET'])
# def get_genre_text():
#     if(last_selected_file==''):
#         return "file_not_uploaded"
#     else:
#         X_test=input_process.process('uploads/metal.00000.wav')
#         prediction=model.predict(X_test)
#         predicted_genre=predict_genre.index(prediction)
#         print("Predicted genre is: ",predicted_genre)
#         return f"{predicted_genre}"
      


if __name__ == '__main__':
    app.run(debug=True)
