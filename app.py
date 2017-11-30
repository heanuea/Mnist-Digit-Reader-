from flask import Flask, render_template, request
from scipy.misc import imsave , imread, imresize
import numpy as np
import keras.models
import re
import base64


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    parseImg(request.get_data())

    
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))

  
    x = x.reshape(1,28,28,1)

    model = models.load_model("model/mnist_model.h5.index")

    
    out = model.predict(x)
    print(out)
   
    response = np.array_str(np.argmax(out, axis=1))
    print(response)
    return response


def parseImg(imgData):
  
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug = True)