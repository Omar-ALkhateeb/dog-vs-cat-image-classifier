from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Dense
new_model = load_model("basic_cnn_20_epochs2.h5")


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('pred9.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = new_model.predict(test_image)
#training_set.class_indices
print(result[0][0])
print(new_model.predict_classes(test_image, verbose=1))
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
