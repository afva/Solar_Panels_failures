import os
from cnn import CNN
from transform_color_space import transform_images
import warnings
warnings.filterwarnings("ignore")


#Global variables
training_dir = 'strings/train'
validation_dir = 'strings/validation'

#TRAIN
cnn = CNN()
bm = 'ResNet50'
epochs = 70
lr = 1e-2
beta_1 = 0.3
cnn.train(training_dir, validation_dir, base_model=bm, epochs=epochs, learning_rate = lr, 
          training_batch_size=64, validation_batch_size=64, beta_1=beta_1, epsilon=1e-5)
cnn.save(f'{bm}_{epochs}_{lr}_{beta_1}')



#PREDICT
print(f'{bm}_{epochs}_{lr}_{beta_1}')
cnn.load(f'{bm}_{epochs}_{lr}_{beta_1}')
# El mejor modelo hasta ahora es el ResNet50_89_0.0001_0.3


#
cnn.predict("training", training_dir, save= True)
cnn.predict("validation", validation_dir,save = True)

