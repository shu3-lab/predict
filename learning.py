from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
import flask

app = flask.Flask(__name__)

def create_model(x,y):
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
    joblib.dump(model, './trained-model/sample-model.pkl')
    return './trained-model/sample-model.pkl'

@app.route("/learning", methods=["POST"])
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
        
        path = create_model(feature, target)
        response["success"] = True
        response["path"] = path
    # return the data dictionary as a JSON response
    return flask.jsonify(response)

if __name__ == "__main__":
    print(" * Flask starting server...")
    app.run()
