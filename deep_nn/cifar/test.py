from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np
model = load_model('cifar_model_cnn.h5')

model.summary()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0


n_to_show = 6
indices = np.random.choice(range(len(x_test)), size=n_to_show, replace=False)

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, f'y = {y_test[idx][0]}',
            size=16,
            ha='center',
            transform=ax.transAxes)
    ax.text(0.5, -0.7, f'pred = {np.argmax(model.predict(img.reshape(1,32,32,3)))}',
            size=16,
            ha='center',
            transform=ax.transAxes)
    ax.imshow(img)
plt.show()