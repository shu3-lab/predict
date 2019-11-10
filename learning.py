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
    #pklをロードする
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
        #特徴量の取得
        if flask.request.get_json().get("feature"):
            feature = flask.request.get_json().get("feature")
        feature = np.array(feature).reshape((len(feature),-1))
        target = []
        #ターゲットの取得
        if flask.request.get_json().get("target"):
            target = flask.request.get_json().get("target")
        target = np.array(target)
        #名前の取得
        name = ''
        if flask.request.get('name'):
            name = flask.request.get('name')

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
        name = flask.request.get('name')
        model = load_model(name)
        input_data_list = []
        if flask.request.get_json().get('data'):
            input_data_list = flask.request.get_json().get('data')
        input_data_list = np.array(input_data_list).reshape((len(input_data_list),-1))
        #ターゲットの取得
        if flask.request.get_json().get("target"):
            target = flask.request.get_json().get("target")
        target = np.array(target)
        #データに対する精度取得
        predict = model.predict(input_data_list)
        trained_accuray = accuracy_score(target, predict)
    
        response["success"] = True
        response["accuray"] = trained_accuray
    return flask.jsonify(response)


if __name__ == "__main__":
    print(" * Flask starting server...")
    app.run()
