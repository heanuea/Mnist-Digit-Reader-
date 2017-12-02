from flask import Flask, render_template, request, jsonify

from scipy.misc import imsave , imread, imresize
import numpy as np
from PIL import Image
import re
import base64

import tensor as ten

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



    
    out = model.predict(x)
    print(out)
   
    response = np.array_str(np.argmax(out, axis=1))
    print(response)
    return response

def get_image(): 
    guess = 0
    if request.method== 'POST':
        img_size = 28, 28 
        image_url = request.values['imageBase64']  
        image_string = re.search(r'base64,(.*)', image_url).group(1)  
        image_bytes = io.BytesIO(base64.b64decode(image_string)) 
        image = Image.open(image_bytes) 
        image = image.resize(img_size, Image.ANTIALIAS)  
        image = image.convert('1') 
        image_array = np.asarray(image)
        image_array = image_array.flatten()  
        
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('LSTM/tensor_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('tmp/'))

            predict_number = tf.argmax(ten.y, 1)
            predicted_number = ten.sess.run([predict_number], feed_dict={ten.x: [image_array]})
            guess = predicted_number[0][0]
            guess = int(guess)
            print(guess)

        return jsonify(guess = guess) 

    return render_template('index.html', guess = guess)


if __name__ == '__main__':
    app.run(debug = True)
