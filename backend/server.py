from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

clf = joblib.load('ChurnRiskScorePrediction.pkl')



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}


# Load your model and preprocess your data
  # Load your model here


def predict_churn_score(data):
    try:
        # Define the list of attributes to exclude
        exclude_attributes = ['Name', 'referral_id', 'churn_risk_score']

        # Create the input_data array by iterating through the attributes
        input_data = [data[attr] for attr in data if attr not in exclude_attributes]

        # Make predictions using the loaded model
        churn_score = clf.predict([input_data])

        return {'churn_score': int(churn_score[0])}
    except Exception as e:
        return {'error': str(e)}


# @app.route('/predict_churn_score', methods=['POST'])
# def predict_churn_score_api():
#     try:
#         data = request.get_json()
#         result = predict_churn_score(data)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)})

@app.route('/predict_churn_score', methods=['POST'])
def predict_churn_score():
    try:
        data = request.get_json()
       
        print(data)
        # Process the input data as needed
        # input_data = list(data.values())
        # feature_order = [
        #  "customer_id","age", "gender", "security_no", "region_category", "membership_category",
        # "joining_date", "joined_through_referral",  "preferred_offer_types",  
        # "medium_of_operation", "internet_option", "last_visit_time", "days_since_last_login",
        # "avg_time_spent", "avg_transaction_value", "avg_transaction_value", "avg_frequency_login_days",
        # "points_in_wallet", "used_special_discount", "offer_application_preference", "past_complaint",
        # "complaint_status", "feedback"]

        
        input_data = [data['customer_id'] ,data['age'],data['gender'],data['security_no'],data['region_category'],data['membership_category'],
                      data['joining_date'],data['joined_through_referral'],data['preferred_offer_types'],data['medium_of_operation'],data['internet_option'],
                      data['last_visit_time'],data['days_since_last_login'],data['avg_time_spent'],data['avg_transaction_value'],data['avg_transaction_value'],
                      data['avg_frequency_login_days'],data['points_in_wallet'],data['used_special_discount'],data['offer_application_preference'],data['past_complaint'],data['complaint_status'],data['feedback']]  # Update with your attribute names
        # input_data = [data[feature] for feature in feature_order]
        # input_data_2d = [input_data]
        
        # print("input_data",input_data_2d)
        # Make predictions using the loaded model
        churn_score = clf.predict([input_data])
        print("\n chur",churn_score)
        return jsonify({'churn_score': churn_score })
    except Exception as e:
        return jsonify({'error': str(e)})

# @app.route('/predict_churn_score', methods=['POST'])
# def predict_churn_score_api():
#     try:
#         data = request.get_json()
#         result = predict_churn_score(data)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)