Creating an executable code for a hybrid blockchain with deep learning for medical records management using federated learning applied on an attention-based method (transformers) and blockchain involves several steps and integrating multiple technologies.
Federated Learning Setup: Distributing the training of the model across multiple nodes (clients) without sharing raw data.
Attention-Based Model (Transformers): Using transformers for medical records analysis.
Blockchain Integration: Ensuring data integrity and secure communication between federated nodes using blockchain.
I'll provide a simplified example with Python, utilizing libraries like TensorFlow for federated learning and transformers, and a basic blockchain implementation. Note that this is a complex and extensive task; the provided code is a simplified starting point and not a complete solution.
Prerequisites
Python 3.8+
TensorFlow
Transformers library (Hugging Face)
Flask (for a simple blockchain network)
flask_cors for handling CORS in Flask
PyCryptodome (for cryptographic functions in blockchain)
Install necessary libraries:
bash

pip install tensorflow transformers flask flask_cors pycryptodome
Federated Learning with Transformers
Here is a simplified version of federated learning with a transformer model:
python

# federated_learning.py

import tensorflow as tf
import tensorflow_federated as tff
from transformers import TFAutoModel, AutoTokenizer

# Define the model
def create_transformer_model():
    transformer_model = TFAutoModel.from_pretrained('bert-base-uncased')
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
    
    embeddings = transformer_model([input_ids, attention_mask])[0]
    output = tf.keras.layers.Dense(1, activation='sigmoid')(embeddings[:, 0, :])
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Wrap the model for federated learning
def model_fn():
    keras_model = create_transformer_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name="input_ids"),
                    tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name="attention_mask")),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Simulate federated data
def preprocess(dataset):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_map(example):
        tokenized = tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'label': example['label']}
    
    return dataset.map(tokenize_map)

# Example federated dataset
client_data = [
    preprocess(tf.data.Dataset.from_tensor_slices({'text': ['patient data 1', 'patient data 2'], 'label': [0, 1]})),
    preprocess(tf.data.Dataset.from_tensor_slices({'text': ['patient data 3', 'patient data 4'], 'label': [1, 0]}))
]

federated_data = [tff.simulation.datasets.ClientData.from_clients_and_fn(client_ids, preprocess) for client_ids in client_data]

# Federated averaging
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, federated_data)
    print(f'Round {round_num}, Metrics: {metrics}')
Basic Blockchain Implementation
A simplified blockchain implementation using Flask:
python

# blockchain.py

import hashlib
import json
from time import time
from uuid import uuid4
from flask import Flask, jsonify, request
from flask_cors import CORS

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.new_block(previous_hash='1', proof=100)
    
    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block
    
    def new_transaction(self, sender, recipient, amount):
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1
    
    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    def proof_of_work(self, last_proof):
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

app = Flask(__name__)
CORS(app)
node_identifier = str(uuid4()).replace('-', '')
blockchain = Blockchain()

@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)
    blockchain.new_transaction(sender="0", recipient=node_identifier, amount=1)
    block = blockchain.new_block(proof)
    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



Running the Code
Run the federated learning script:
bash

python federated_learning.py
Start the blockchain server:
bash

python blockchain.py
Integration
The integration part would involve setting up secure communication channels between federated learning clients and the blockchain server, ensuring data integrity and authenticity. Each client would use the blockchain to log their updates and model weights without exposing raw medical data.
This example provides a foundation for building a more comprehensive and secure system for managing medical records using hybrid blockchain and deep learning techniques.

Which Attention-Based Model was used ? 
The provided code uses the BERT (Bidirectional Encoder Representations from Transformers) model as the attention-based model for medical records analysis.
BERT is a popular transformer model pre-trained on a large corpus of text, making it suitable for various natural language processing tasks, including those involving medical records.
A breakdown of how BERT is used in the provided code:
Model Definition: The TFAutoModel.from_pretrained('bert-base-uncased') line loads the pre-trained BERT model from the Hugging Face transformers library.
Input Handling: The model takes input_ids and attention_mask as inputs, which are typical for transformer models like BERT. The input_ids represent the tokenized text, and the attention_mask indicates which tokens should be attended to (1) and which should be ignored (0).
Output Layer: The embeddings from BERT are passed through a dense layer to produce the final output. In this example, the dense layer has a single output unit with a sigmoid activation function for binary classification.

Relevant part of the code for the attention-based model:
python

from transformers import TFAutoModel, AutoTokenizer

# Define the model
def create_transformer_model():
    transformer_model = TFAutoModel.from_pretrained('bert-base-uncased')
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
    
    embeddings = transformer_model([input_ids, attention_mask])[0]
    output = tf.keras.layers.Dense(1, activation='sigmoid')(embeddings[:, 0, :])
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
This setup uses BERT as the backbone for processing medical records and leverages its attention mechanism to focus on relevant parts of the input text, which is crucial for tasks involving large and complex text data like medical records.

