# importing the lib
from flask import Flask , render_template, request
import joblib
app = Flask(__name__)

#load the model
model = joblib.load('Rev_Iris_Modl.pkl')

@app.route('/')
def home():
    return render_template('Rev_Iris1.html')

@app.route('/data', methods=['post'])
def data():
    #Id = request.form.get('Id')
    SepalLengthCm = request.form.get('SepalLengthCm')
    SepalWidthCm = request.form.get('SepalWidthCm')
    PetalLengthCm = request.form.get('PetalLengthCm')
    PetalWidthCm = request.form.get('PetalWidthCm')
    #Species = request.form.get('Species')
    #pedi = request.form.get('pedi')
    #age = request.form.get('age')

    result = model.predict([[ SepalLengthCm , SepalWidthCm, PetalLengthCm, PetalWidthCm]])
#pip 
    if result[0]==1:
        #data = 'Iris Dataset working'
        data = 'result1'
    else:
        #data = 'Iris Not Dataset working'
        data = result

    #print(data)
    #return 'data received'
    return render_template('Rev_Iris1_Pred.html', data=data)

app.run(host = '0.0.0', port=8080) # should be always at the end
#app.run(debug = True) # should be always at 