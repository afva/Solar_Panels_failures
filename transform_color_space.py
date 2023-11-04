# change all images in dataset_mod to different color spaces
import os
import cv2
import matplotlib.pyplot as plt

def transform_images():

# Read images
    for tipo in ['training', 'validation','test']:
        for defecto in ['con_defecto', 'sin_defecto']:
            for root, dirs, files in os.walk(r"dataset_mod\{}\{}_{}".format(tipo, tipo, defecto)):
                
                for name in files:
                    print(name)
                    img_path = os.path.join(root, name)
                    img = cv2.imread(img_path)
                    img = cv2.Canny(img, 100, 200)
                    cv2.imwrite(img_path, img)
                    # ax.imshow(img, cmap='gray')
                    plt.show()
                