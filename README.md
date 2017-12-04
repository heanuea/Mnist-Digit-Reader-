# **_About Me_**

My name is Alan Heanue I am a 4th year software development Studying in the Galway-Mayo Intitute of Technology
This is one of 5 modules i am Working on this semester it focuses mainly on emerging technologies and how these technologies can be used for industry based usage

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


**Ajax call**

```Ajax 

              $.ajax({
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

**Flask**

I needed Flask to perform three tasks besides rendering the website: collect the base64-encoded handwritten digit images; hand the images over to the machine learning recognizer; and return the result back to the front end. The complete Flask code is shown here below.

```python
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
Two functions implementing the Flask tasks follow. The functions use Flasks's route() decorator to indicate what URL should trigger the function. While index() simply renders the web page, get_image() handles all the digit recognition work. It collects the data, preprocesses the digit images, and runs the digit recognizer. This uses functionality implemented in the PIL module, addressed in more detail in the sections discussing data preprocessing and the machine learning classifier below. Additionally, get_image() also prepares the output by identifying the class (digits between 0 and 9) with the highest probability and composing the string to be displayed to the user. If the determined highest probability is lower than 60%, the output class is not shown and the string is a message indicating the digit could not be identified. This is to handle the fact that the model will always output a digit class, even if the drawing does not look at all like a digit. Otherwise, the string contains the predicted digit and its probability.
**MNIST**
As I mentioned above, handwritten digit recognition is a widely studied problem in the field of computer vision. A popular training dataset, the [MNIST](http://yann.lecun.com/exdb/mnist/), has been around for quite some time and has been used extensively to benchmark developments in the field. It is a subset of a larger dataset distributed by the National Institute of Standards and Technology (NIST).[MNIST](http://yann.lecun.com/exdb/mnist/) consists of scanned grayscale digital images of handwritten digits, including balanced sets of 60,000 and 10,000 training and test images, respectively. The images have been size-normalized and centered, making up a nice and clean dataset ready for use. Here are some example.

[![N|Solid](https://github.com/heanuea/Mnist-Digit-Reader--master/Images/Figure-13-Scatter-SVM-non-support-vectors-on-MNIST-data.png)]

**TensorFlow**

I decided to use TensorFlow, Google's machine learning software library to implement the machine learning model. To make the implementation even simpler, I went one step higher in the abstraction level and used TFlearn, a software library providing "a higher-level API to TensorFlow". As for the choice of machine learning algorithm, the best classification accuracies are achieved with deep convolutional neural networks (CNNs), as you can see in the list of research results on MNIST's webpage or in this other curated list. However, you can get very decent accuracies with relatively shallow CNNs too. So, at least for the first implementation, I decided to use a relatively simple CNN architecture.
```python

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])  
    #We now define the weights W and biases b for our model. 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


y = tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for _ in range(10000):
    batch = mnist.train.next_batch(100)  
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```
 interactive session also sets the created session as the default session, so you don't need to precede the session level commands like eval with the sess reference. We need some placeholders for the MNIST data we are passing and training. As X is the raw image data, and Y bar is the digit zero through nine of the image. And we use none because we don't know how many images we're going to pass in. We define the weights and biases to our model in a layer. Before Variables can be used within a session, they must be initialized using that session,
 After that we set the training. TensorFlow has a variety of built-in optimization algorithms. For this example, we will use steepest gradient descent,
 ```python

 train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 ```
 with a step length of 0.5, to descend the cross entropy.

 **PREPROCESSING THE HANDWRITTEN DIGIT IMAGES**

The data sent to the server from the front end needs to be peprocessed before classification. The first step is to convert the base64-encoded image of the digit drawn by the user to a NumPy ndarray data structure with an URl the Function image_url = request.values['imageBase64'] , at which point the data could be fed to the classifier. However, in order to minimize the loss in classificatin accuracy caused by the differences to the MNIST training data, a few extra steps need to be taken as we seen in the Flask App above this is converted to PIL Image and it tries to resize it using image = image.resize(img_size, Image.ANTIALIAS) for tensorflow. 


## **_Requirements_**

- Python 3.5.0 
- Flask 
- Tensorflow 
You will need to install a few packages for this to work to do this you can use this cmd 
```
pip install "package"
```
## **_To run this app_**
- git clone this app or just download zip 
```
git clone ssh://john@example.com/path/to/my-project.git
```
- go into project 
```
cd my-project
```
- and to run 
```
python app.py 
```

### **_Conclusion_**
Overall i like learning abou this subject i understand the fundametals of Machine learning (Tensorflow), i discovered that machine learning is not as complex as it may it seems with the tools and understanding of models and training them. With the Project i wish i had time to do a better job and have multilayer convolutionalin my project  and i wanted to implement [TFlearn](http://tflearn.org/) i didnt use [Keras](https://keras.io) as i wanted to understand tensorflow in more depth and see how it worked in terms of how each step takes i found Keras puts alot of the tensorflow into methods so does alot of the work for ya. But i will look over that on myspare time, this topic was great because not only is it new but it going to be massively used in the future. 

### **_References_**
 
 - http://tflearn.org/
 - https://www.tensorflow.org/get_started/mnist/pros
 - https://app.pluralsight.com/library/courses/tensorflow-getting-started/transcript
 - https://github.com/msrks/DL_num-pred
 - http://luisvalesilva.com/datasimple/digitre.html
 - https://github.com/luisvalesilva/digitre/blob/master/runtime.txt
 - https://guillaumegenthial.github.io/serving.html
