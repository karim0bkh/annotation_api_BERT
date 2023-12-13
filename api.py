import subprocess
from flask import Flask, request, jsonify
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import uuid
import os
import json
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:karim@localhost:5432/annotation'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    model_info = db.Column(db.JSON, nullable=True) 
    model_id = db.Column(db.UUID, nullable=False)

class Files(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.BigInteger)
    file_name = db.Column(db.String)
    flag = db.Column(db.Integer)
    annotations = db.Column(db.JSON)

models_info = {}

def create_model():
    folder_name = str(uuid.uuid4())
    output_folder = os.path.join('models', folder_name)
    os.makedirs(output_folder)

    models_info[folder_name] = {'status': 'created'}

    return folder_name

@app.route('/create_model', methods=['POST'])
def create_model_endpoint():
    try:
        folder_name = create_model()
        user_id = request.json.get('user_id')

        # Save model information to the database
        new_model_data = TrainingData(
            user_id=user_id,
            model_id=folder_name,
            model_info=json.dumps(models_info[folder_name])
        )
        db.session.add(new_model_data)
        db.session.commit()

        return jsonify({'status': 'success', 'folder_name': folder_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    try:
        data = request.json
        user_id = data.get('user_id')

        # Fetch the model folder name based on user_id
        model_data = TrainingData.query.filter_by(user_id=user_id).first()

        if model_data is None:
            return jsonify({'error': 'Model not found for the specified user_id.'}), 404

        folder_name = model_data.model_id
        output_folder = os.path.join('models', str(folder_name))

        nlp = spacy.load("en_core_web_lg")

        # Fetch files based on user_id and flag
        files = Files.query.filter_by(user_id=user_id, flag=1).all()
        print(files[0].annotations)
        training_data = []
        for file_data in files:
            temp_dict = {'text': file_data.annotations['content'], 'entities': []}
            for annotation in file_data.annotations['annotations']:
                start = annotation['start']
                end = annotation['end'] + 1
                label = annotation['tag_name'].upper()
                temp_dict['entities'].append((start, end, label))
            training_data.append(temp_dict)



        # Save training data to the database
        
        doc_bin = DocBin()
        for training_example in training_data:
            text = training_example['text']
            labels = training_example['entities']
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    ents.append(span)
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents
            doc_bin.add(doc)

        doc_bin.to_disk(os.path.join(output_folder, "train.spacy"))

        init_command = "python3 -m spacy init fill-config base_config.cfg config.cfg"
        train_command = f"python3 -m spacy train config.cfg --output {output_folder} --paths.train ./{output_folder}/train.spacy --paths.dev ./{output_folder}/train.spacy --gpu-id 0 --verbose"

        subprocess.run(init_command, shell=True, check=True)
        subprocess.run(train_command, shell=True, check=True)

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    try:
        data = request.json
        user_id = data.get('user_id')

        # Fetch the model folder name based on user_id
        model_data = TrainingData.query.filter_by(user_id=user_id).first()

        
        folder_name = model_data.model_id
        output_folder = os.path.join('models', str(folder_name))
        nlp_ner = spacy.load(os.path.join(output_folder, "model-best"))

        text = data.get('text', '')

        doc = nlp_ner(text)

        json_obj = doc.to_json()

        return jsonify({'json': json_obj})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
