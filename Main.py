#from platform import libc_ver


#importing the lib
from flask import Flask, render_template


app = Flask(__name__)
@app.route('/')

def home():
    return render_template('home.html')

@app.route('/images')
def images():
    return "This is Image page"

@app.route('/Content')
def content():    
    return "This is Content page"


app.run() # This should be always at the end

"""
http: hyper text transfer protocol
127.0.0.1 - ip address - localhost
:5000 - port
/ - route

http://127.0.0.1:5000/

"""