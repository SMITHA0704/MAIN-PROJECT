from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   exp1=int(request.values['Year'])
   exp2= int(request.values['State'])
   exp3= int(request.values['City'])
   exp4= np.sin(float(request.values['Month'])*(2.*np.pi/12))
   exp5= np.cos(float(request.values['Month'])*(2.*np.pi/12))
   output=model.predict([[exp1,exp2,exp3,exp4,exp5]])
   output=output.item()
   output=round(output)
   print(output)
   if output <= 50:
        return render_template ('result.html',prediction_text="Air quality-Good!! AQI value is {}".format(output))
   elif output > 50 and output <= 100:
        return render_template ('result.html',prediction_text="Air quality-Satisfactory!! AQI value is {}".format(output))
   elif output > 100 and output <= 200:
        return render_template ('result.html',prediction_text="Air quality-Moderate !! AQI value is {}".format(output))
   elif output > 200 and output <= 300:
        return render_template ('result.html',prediction_text="Air quality-Poor !! AQI value is {}".format(output))
   elif output > 300 and output <= 400:
        return render_template ('result.html',prediction_text="Air quality-Very Poor!! AQI value is {}".format(output))
   elif output > 400:
        return render_template ('result.html',prediction_text="Air quality-Severely Poor!! AQI value is {}".format(output))
if __name__=='__main__':
    app.run(port=8000)
