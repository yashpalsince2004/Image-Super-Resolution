import tensorflow as tf
from keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

print("✅ TensorFlow version:", tf.__version__)

# 📄 Load a sample image
img_path = "lena.jpg"

# If lena.jpg is not present, download it
if not os.path.exists(img_path):
    import urllib.request
    print("📥 Downloading lena.jpg …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        img_path
    )

# ✅ Read image
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # Y channel
img = cv2.resize(img, (128, 128))

low_res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
upscaled = cv2.resize(low_res, (128, 128), interpolation=cv2.INTER_CUBIC)

# 🔷 Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(low_res, cmap='gray')
plt.title('Low-Res')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(upscaled, cmap='gray')
plt.title('Upscaled (Input)')
plt.axis('off')
plt.show()


# 🧠 Build SRCNN Model
def build_srcnn():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(128, 128, 1)))
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (5, 5), activation='linear', padding='same'))
    model.compile(optimizer='adam', loss='mse')
    return model


srcnn = build_srcnn()
srcnn.summary()

# 📊 Prepare Data
x_train = upscaled / 255.0
y_train = img / 255.0

x_train = np.expand_dims(x_train, axis=(0, -1))
y_train = np.expand_dims(y_train, axis=(0, -1))

# 🎯 Train SRCNN
srcnn.fit(x_train, y_train, epochs=100, verbose=2)

# 🔮 Predict & Visualize
pred = srcnn.predict(x_train)
pred = np.clip(pred[0, :, :, 0] * 255.0, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(upscaled, cmap='gray')
plt.title('Upscaled')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(pred, cmap='gray')
plt.title('SRCNN Output')
plt.axis('off')
plt.show()

# 🔷 PSNR
print(f"Upscaled PSNR: {psnr(img, upscaled):.2f} dB")
print(f"SRCNN PSNR: {psnr(img, pred):.2f} dB")

# Save the model
srcnn.save('srcnn_model.h5')
print("✅ SRCNN model saved as 'srcnn_model.h5'")