import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


# Create flask app
flask_app = Flask(__name__)

# Load the joblib file
model = joblib.load('dt.joblib')


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():

    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    # Make a prediction on new data
    new_data = [[float(num) for num in arr] for arr in features]
    new_data = pd.DataFrame(new_data, columns=['Loudness', 'Valence', 'Danceability', 'Acousticness',
                                               'Instrumental', 'Audio_class', 'Sentiment_class', 'Audio + Lyrics analysis', 'Total_mental_health'])
    # return render_template("index.html")

    if new_data.loc[0, 'Instrumental'] <= 0:
        new_data.loc[0, 'Instrumental'] = 1e-7

    # apply the logarithm transformation
    new_data['Instrumental'] = np.log(new_data['Instrumental'])
    # Scale the input data using the same mean and standard deviation from X_train

    #  convert the data types of specific columns using astype()
    new_data[['Audio_class', 'Sentiment_class', 'Audio + Lyrics analysis']] = new_data[[
        'Audio_class', 'Sentiment_class', 'Audio + Lyrics analysis']].astype('int64')

    new_data = np.delete(new_data, 6, axis=1)

    prediction = model.predict(new_data)

    prediction = prediction[0]
    print(prediction)

    if (prediction == 0):
        prediction = 'low'
    elif (prediction == 1):
        prediction = 'med'
    elif (prediction == 2):
        prediction = 'high'

    return render_template("index.html", p_class=prediction)


if __name__ == "__main__":
    flask_app.run(debug=True)
