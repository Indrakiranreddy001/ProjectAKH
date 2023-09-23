from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import string


app = Flask(__name__)
cors = CORS(app, resources={r"/ProjectAKH/generate-dataset": {"origins": "*"}})


@app.route('/generate-dataset', methods=['POST'])
def generate_dataset():
    data = request.json
    features = data['features']
    data_types = data['dataTypes']
    sample_data = data['sampleData']

    dataset = generate_synthetic_dataset(features, data_types, sample_data)

    return jsonify(dataset)

def generate_synthetic_dataset(features, data_types, sample_data):
    dataset = []

    for _ in range(sample_data):
        data_row = {}

        for feature, data_type in zip(features, data_types):
            if data_type == 'string':
                data_row[feature] = ''.join(random.choice(string.ascii_letters) for _ in range(10))
            elif data_type == 'number':
                data_row[feature] = random.randint(0, 100)
            # Add more data types as needed

            # Handle other data types here...

        dataset.append(data_row)

    return dataset

if __name__ == '__main__':
    app.run(debug=True,port=5010)
