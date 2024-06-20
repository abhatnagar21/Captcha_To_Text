pip install captcha
import os
import cv2
import numpy as np
import random
import string
from captcha.image import ImageCaptcha
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, InputLayer
from tensorflow.keras.utils import to_categorical
def random_text(length=5):
    letters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_captcha_images(num_images=1000, output_dir='captchas'):
    os.makedirs(output_dir, exist_ok=True)
    image = ImageCaptcha()

    for i in range(num_images):
        text = random_text()
        image.write(text, os.path.join(output_dir, f'{text}.png'))

generate_captcha_images()
def preprocess_image(image_path, img_width=100, img_height=40):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)
    return img

def load_data(data_dir, img_width=100, img_height=40):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(data_dir, filename)
            X.append(preprocess_image(image_path, img_width, img_height))
            y.append(filename.split('.')[0])  # Extract label from filename

    return np.array(X), np.array(y)

X, y = load_data('captchas')
# 3. Encode labels
captcha_length = 5
num_classes = 36
char_list = string.ascii_uppercase + string.digits
char_to_index = {char: idx for idx, char in enumerate(char_list)}

def encode_labels(labels, captcha_length, num_classes):
    encoded = np.zeros((len(labels), captcha_length, num_classes), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            encoded[i, j, char_to_index[char]] = 1
    return encoded

y_encoded = encode_labels(y, captcha_length, num_classes)
# 4. Create and compile model
def create_model(input_shape, captcha_length, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(captcha_length * num_classes, activation='softmax'))
    model.add(Reshape((captcha_length, num_classes)))
    return model

input_shape = (40, 100, 1)
model = create_model(input_shape, captcha_length, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 5. Train the model
model.fit(X, y_encoded, epochs=30, batch_size=32, validation_split=0.2)
# 6. Decode new CAPTCHA
def decode_captcha(model, image_path, char_list, char_to_index):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    decoded_text = ''.join([char_list[np.argmax(char)] for char in prediction[0]])
    return decoded_text

index_to_char = {idx: char for char, idx in char_to_index.items()}

captcha_image_path = '0121Q.png'
decoded_text = decode_captcha(model, captcha_image_path, index_to_char, char_to_index)
print(f'Decoded CAPTCHA text: {decoded_text}')
