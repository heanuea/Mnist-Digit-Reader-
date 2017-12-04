# taken from https://community.canvaslms.com/thread/2595

from flask import Flask, render_template,url_for, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import re
import io
import base64

import tensor as ten

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def get_image(): 
    guess = 0
    if request.method== 'POST':
        #requests image from url 
        img_size = 28, 28 
        image_url = request.values['imageBase64']  
        image_string = re.search(r'base64,(.*)', image_url).group(1)  
        image_bytes = io.BytesIO(base64.b64decode(image_string)) 
        image = Image.open(image_bytes) 
        image = image.resize(img_size, Image.LANCZOS)  
        image = image.convert('1') 
        image_array = np.asarray(image)
        image_array = image_array.flatten()  
        
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('tmp/tensor_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('tmp/'))

            predict_number = tf.argmax(ten.y, 1)
            predicted_number = ten.sess.run([predict_number], feed_dict={ten.x: [image_array]})
            guess = predicted_number[0][0]
            guess = int(guess)
            print(guess)

        return jsonify(guess = guess) #returns as jason format

    return render_template('index.html', guess = guess)


if __name__ == '__main__':
    app.run(debug = True)
