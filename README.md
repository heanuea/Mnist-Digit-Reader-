# **_About Flask Python Digit Reader_**

The web app is written in Python using the Flask web framework. I use JavaScript to collect drawings in an HTML canvas element and Machine Learning (ML) for handwritten digit recognition. The digit recognizer is a Convolutional Neural Network (CNN) trained on the MNIST dataset using the Tensorflow software library.

After training, the CNN showed 92.92% accuracy on MNIST's test dataset. To learn all about my architecture checkout my Diagram about it below. 





## **_Overview_**
**_[Project Brief](https://emerging-technologies.github.io/problems/project.html)_**


[![N|Solid](https://cldup.com/IU6Qs_rV6q.jpg)](https://cldup.com/IU6Qs_rV6q.jpg/nsolid)

I will cover a much simpler approach, similar to the one used here. I’ll use

- [Flask](flask.pocoo.org/)to build the API (back-end)
- jquery.[ajax](http://api.jquery.com/jquery.ajax/) to handle requests to the API from your client (front-end)
- I use JavaScript[(FabricJs)](http://fabricjs.com/) to allow the user to draw graphics on the fly in an HTML canvas element

The JavaScript code is adapted from a script I found on CodePen. The image drawn by the user is collected and encoded in base64 format for convenient processing. 
Base64 encoding is a way to represent binary data in an ASCII string format, often used for images in web development.

### **_Steps Taken_**
The next step is to share the digit with the server, where Flask will handle the rest. I am using AJAX with JQuery, which allows me to update the page to display the result of the digit recognition without having to reload the whole web page. To write the code I used the example in Flask's documentation as a starting point and changed it to transfer the data in JSON format in the body of a POST request. The following is the final JavaScript function (found at the end of the source html in the page index.html).


**Step 1**
Ajax call

```                $.ajax({
                    type: 'post',
                    url: '/',
                    data: {
                        imageBase64 : imgURL
                    },
                
                    success: function(data){
                        $('#guess').text(data.guess);
                    }              
                });                     
            }
```

**_FLASK CODE_**

I needed Flask to perform three tasks besides rendering the website: collect the base64-encoded handwritten digit images; hand the images over to the machine learning recognizer; and return the result back to the front end. The complete Flask code is shown here below.

```from flask import Flask, render_template,url_for, request, jsonify
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
        image = image.resize(img_size, Image.ANTIALIAS)  
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

```

## **_Technologies_**
### **_Python_**

### **_Flask_**

### **_TensorFlow_**
