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
## **_Technologies_**
### **_Python_**

### **_Flask_**

### **_TensorFlow_**
