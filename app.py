from flask import Flask, render_template, request
import iris_model
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction
    if output == 0:
        return 'Iris-setosa'
    elif output == 1:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'

    return render_template('index.html', prediction_text= 'OUTPUT {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)