from flask import Flask, request, jsonify
from src.model.temp_predict import get_temp

app = Flask(__name__)

@app.route("/")
def homepage():
    return "Welcome to the Keras REST API!"

@app.route('/temp_predict', methods=['POST'])
def temp_predict():
    data = request.json
    # print(data)
    try:
        result  = get_temp(data['day'], data['month'], data['year'],data['type'])
        if isinstance(result, str):
            return jsonify("Previous date not found in data store!!!")
        else:
            return  jsonify(result.tolist())
    except (TypeError) as e:
        raise TypeError("Json not serializable")

