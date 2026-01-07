import numpy
from flask import Flask,render_template,request
import pickle
app = Flask(__name__)
app.name = 'Hyderabad House Price Prediction'

model = r"D:\python\hydhouse\multi_xgb_model"
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

        return render_template(
            "result.html",
            prediction_price =f"Predicted Price: {price:.2f} Lakh",predicted_rate_per_sqft = f"Rate per sqft: {rate_persqft:.2f}"
        )
    except Exception as e:
        return render_template(
            "result.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
