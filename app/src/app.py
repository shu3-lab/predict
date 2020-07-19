from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
import flask
import os
from sklearn.metrics import accuracy_score


app = flask.Flask(__name__)

def create_model(x,y,name):
    print(x)
    print(y)
    
    #データを標準化する
    sc = StandardScaler()
    sc.fit(x)
    x_sc = sc.transform(x)

    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)
    #モデルの学習を行う
    model.fit(x_sc, y)
    #学習済みモデルの保存
    joblib.dump(model, './trained-model/' + name + '.pkl')
    return os.path.abspath(name + ".pkl")

def load_model(name):
    #load a PKI file
    loaded_model = joblib.load('./trained-model/' + name + '.pkl')
    return loaded_model

@app.route("/learning/model", methods=["POST"])
def learn():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "POST":
        feature = []
        #get feature values
        if flask.request.get_json().get("feature"):
            feature = flask.request.get_json().get("feature")
        feature = np.array(feature).reshape((len(feature),-1))
        target = []
        #get target values
        if flask.request.get_json().get("target"):
            target = flask.request.get_json().get("target")
        target = np.array(target)
        #get name
        name = ''
        if flask.request.args.get('name'):
            name = flask.request.args.get('name')
        else:
            # validate name
            response["success"] = False
            response["path"] = "none"
            response["message"] = "The value of name is required!"
            return flask.jsonify(response)

        path = create_model(feature, target, name)
        response["success"] = True
        response["path"] = path
    # return JSON response
    return flask.jsonify(response)

@app.route("/learning/evalution", methods=["GET"])
def eval():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "GET":
        if flask.request.args.get('name'):
            name = flask.request.args.get('name')
        else:
            # validate name
            response["success"] = False
            response["path"] = "none"
            response["message"] = "The value of name is required!"
            return flask.jsonify(response)
        
        model = load_model(name)
        input_data_list = []
        if flask.request.get_json().get('data'):
            input_data_list = flask.request.get_json().get('data')
        input_data_list = np.array(input_data_list).reshape((len(input_data_list),-1))
        #get target values
        if flask.request.get_json().get("target"):
            target = flask.request.get_json().get("target")
        target = np.array(target)
        #get accuracy of prediction by input data
        predict = model.predict(input_data_list)
        trained_accuray = accuracy_score(target, predict)
    
        response["success"] = True
        response["accuray"] = trained_accuray
    return flask.jsonify(response)

@app.route("/")
def index():
    #return a index message.
    response = {
        "message" : "Prediction API is runnning!"
    }
    return flask.jsonify(response)

if __name__ == "__main__":
    print(" * Flask starting server...")
    app.run()
