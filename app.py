import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np

scaler = pickle.load(open('scaler.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def employee():
    return render_template('employee.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    features_list = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender','EverBenched', 'ExperienceInCurrentDomain']

    education = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
    city = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
    gender = {'Male': 0, 'Female': 1}
    ever_bench = {'No': 0, 'Yes': 1}

    user_features = np.zeros(len(features_list))
    user_features[features_list.index('Education')] = education[data['Education']]
    user_features[features_list.index('JoiningYear')] = data['Joining Year']
    user_features[features_list.index('City')] = city[data['City']]
    user_features[features_list.index('PaymentTier')] = data['Payment Tier']
    user_features[features_list.index('Age')] = data['Age']
    user_features[features_list.index('Gender')] = gender[data['Gender']]
    user_features[features_list.index('EverBenched')] = ever_bench[data['Ever Benched']]
    user_features[features_list.index('ExperienceInCurrentDomain')] = data['Experience In Current Domain']

    new_data = scaler.transform(np.array(user_features).reshape(1,-1))
    output = rf_model.predict(new_data)

    if output[0] == 0:
        return "This Employee will not leave company"
    else:
        return "This Employee is likely to leave company"


if __name__ == '__main__':
    app.run(debug = True)