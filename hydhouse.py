import numpy
from flask import Flask,render_template,request
import pickle
import pandas as pd
app = Flask(__name__)
app.name = 'Hyderabad House Price Prediction'

model = "multi_xgb_model"
with open(model,'rb') as file:
    multi_xgb_model = pickle.load(file)
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods =['POST'])
def predict():
    try:
        title = int(request.form.get('title'))
        location = int(request.form.get('location'))
        area_insqft = int(request.form.get('area_insqft'))
        building_status = int(request.form.get('building_status'))

        input_data = numpy.array([[title,location,area_insqft,building_status]])
        predictions = multi_xgb_model.predict(input_data)
        price, rate_persqft = predictions[0][0], predictions[0][1]
        title_map_list = pd.read_csv("title_mapping.csv")
        title_map = title_map_list.loc[title_map_list['code'] == title, 'category'].iloc[0]
        building_status_list = pd.read_csv("building_status_mapping.csv")
        building_status_re = building_status_list.loc[building_status_list['code']== building_status,'category'].iloc[0]
        location_list = pd.read_csv("location_mapping.csv")
        location_re = location_list.loc[location_list['code']==location,'category'].iloc[0]



        return render_template(
            "result.html",
            title_map = title_map,
            location_re = location_re,
            area_insqft = area_insqft,
            building_status_re = building_status_re,
            prediction_price =f"Predicted Price: {price:.2f} Lakh",predicted_rate_per_sqft = f"Rate per sqft: {rate_persqft:.2f}"
        )
    except Exception as e:
        return render_template(
            "result.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False)
