from flask import request
app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
        # Preprocess the uploaded image and make predictions
        # Return the predicted class and nutrition values
        return 'Prediction results'
