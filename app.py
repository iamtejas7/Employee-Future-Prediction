import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np

scaler = pickle.load(open('scaler.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def employee():
    return render_template('emp2.html')

# new change

# api for postman testing
# @app.route('/predict_api', methods = ['POST'])
# def predict_api():
#     data = request.json['data']
#     features_list = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender','EverBenched', 'ExperienceInCurrentDomain']

#     education = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
#     city = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
#     gender = {'Male': 0, 'Female': 1}
#     ever_bench = {'No': 0, 'Yes': 1}

#     user_features = np.zeros(len(features_list))
#     user_features[features_list.index('Education')] = education[data['Education']]
#     user_features[features_list.index('JoiningYear')] = data['Joining Year']
#     user_features[features_list.index('City')] = city[data['City']]
#     user_features[features_list.index('PaymentTier')] = data['Payment Tier']
#     user_features[features_list.index('Age')] = data['Age']
#     user_features[features_list.index('Gender')] = gender[data['Gender']]
#     user_features[features_list.index('EverBenched')] = ever_bench[data['Ever Benched']]
#     user_features[features_list.index('ExperienceInCurrentDomain')] = data['Experience In Current Domain']

#     new_data = scaler.transform(np.array(user_features).reshape(1,-1))
#     output = rf_model.predict(new_data)

#     if output[0] == 0:
#         return "This Employee will not leave company"
#     else:
#         return "This Employee is likely to leave company"


# api for template employee.html
# @app.route('/predict', methods = ['POST'])
# def predict():
#     data = request.form
#     input_values = list(data.values())
#     user_features = []
#     for i in input_values:
#         try:
#             user_features.append(eval(i))
#         except:
#             user_features.append(i.strip().lower())
#     # print(user_features)

#     education = {'bachelors': 0, 'masters': 1, 'phd': 2}
#     city = {'bangalore': 0, 'pune': 1, 'new delhi': 2}
#     gender = {'male': 0, 'female': 1}
#     ever_bench = {'no': 0, 'yes': 1}

#     try:
#         user_features[0] = education[user_features[0]]
#         user_features[2] = city[user_features[2]]
#         user_features[5] = gender[user_features[5]]
#         user_features[6] = ever_bench[user_features[6]]
#         # print(user_features)
#     except:
#         return render_template("employee.html", prediction_text = "Please fill all boxes with correct values")

#     else:
#         new_data = scaler.transform(np.array(user_features).reshape(1,-1))
#         output = rf_model.predict(new_data)
#         # print(output)

#         if output[0] == 0:
#             return render_template("employee.html", prediction_text = "This Employee will not leave company")
#         else:
#             return render_template("employee.html", prediction_text = "This Employee is likely to leave the company")


# api for template emp2.html
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        education = request.form['Education']
        joining_year = request.form['JoiningYear']
        city = request.form['City']
        payment_tier = request.form['PaymentTier']
        age = request.form['Age']
        gender = request.form['Gender']
        ever_benched = request.form['EverBenched']
        exp = request.form['ExperienceInCurrentDomain']

        user_input = [education, joining_year, city, payment_tier, age, gender, ever_benched, exp]

        new_data = scaler.transform(np.array(user_input).reshape(1,-1))
        output = rf_model.predict(new_data)

    except:
        return render_template("emp2.html", prediction_text = "Please fill all boxes.")

    else:
        if output[0] == 0:
            return render_template("emp2.html", prediction_text = "This Employee will not leave company")
        else:
            return render_template("emp2.html", prediction_text = "This Employee is likely to leave the company")


if __name__ == '__main__':
    app.run(debug = True)