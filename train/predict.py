import tensorflow as tf 
from keras.preprocessing import image
import numpy as np
import cv2
import imutils
from skimage.transform import resize, pyramid_reduce
from skimage.color import rgb2grey
import matplotlib.pyplot as plt

with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    unsecure_loaded_model = tf.keras.models.model_from_json(loaded_model_json)

unsecure_loaded_model.load_weights("model/model_weights.h5")
unsecure_loaded_model._make_predict_function()   
print(unsecure_loaded_model.summary())

signs = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing','No passing for vechiles over 3.5 metric tons','Right-of-way at the next intersection','Priority road','Yield','Stop','No vechiles','Vechiles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left','Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory','End of no passing','End of no passing by vechiles over 3.5 metric tons']



video = cv2.VideoCapture('video/v1.mp4')
while(True):
    ret, frame = video.read()
    print(ret)
    if ret == True:
        #cv.imwrite("test.jpg",frame)
        #imagetest = image.load_img("test.jpg", target_size = (150,150))
        imagetest = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        #imagetest = rgb2grey(imagetest)
        imagetest = cv2.cvtColor(imagetest, cv2.COLOR_BGR2GRAY)
        #imagetest = (imagetest / 255.0) # rescale
        imagetest = cv2.resize(imagetest, (32, 32), cv2.INTER_AREA) #resize
        #plt.imshow(imagetest, cmap='gray')
        #plt.show()
        #imagetest = cv2.resize(frame, (32,32), cv2.INTER_AREA)
        #imagetest = cv2.cvtColor(imagetest, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("test.jpg",imagetest)
        imagetest = image.img_to_array(imagetest)
        imagetest = np.expand_dims(imagetest, axis = 0)
        predict = unsecure_loaded_model.predict_classes(imagetest)
        print(predict[0])
        msg = signs[predict[0]];
        
        text_label = "{}: {:4f}".format(msg, 80)
        cv2.putText(frame, text_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
