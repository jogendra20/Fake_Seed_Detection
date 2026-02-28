from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
# Load with custom metric support
model = tf.keras.models.load_model('app/seed_model_v15.h5', custom_objects={'f05_score': f05_score})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No image'}), 400
    
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    
    prediction = float(model.predict(img_array)[0][0])
    
    # V15 Logic: Categorize based on strictness
    if prediction > 0.85: status = "Confirmed Counterfeit"
    elif prediction < 0.15: status = "Verified Authentic"
    else: status = "Inconclusive - Manual Review Required"

    return jsonify({
        'version': '15.0.4',
        'probability': prediction,
        'verdict': status,
        'intervention_needed': 0.15 < prediction < 0.85
    })

if __name__ == '__main__':
    app.run(port=5000)
