from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Preprocess training and validation data
def data_generator(path_file):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=10, zoom_range=0.1,
                                 horizontal_flip=True, validation_split=0.2)
    train_data = datagen.flow_from_directory(path_file, (224, 224), subset='training')
    val_data = datagen.flow_from_directory(path_file, (224, 224), subset='validation')
    return train_data, val_data

# Define the model
# Load pretrained mobilenet as base model without including top layers and build the top layers over it
# and train only the top layers using the dataset
def model():
    base_model = MobileNetV2(input_tensor=Input(shape=(224, 224, 3)), include_top=False,
                             weights='source/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    model = AveragePooling2D((7, 7))(base_model.output)
    model = Flatten(name='Flatten')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(2, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=model)
    plot_model(model, to_file='config.png')
    return model, base_model


def train(model, base_model, train_data, val_data):
    tensor_board = TensorBoard(log_dir='logs1')
    monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=10)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    opt = Adam(lr=1e-4)
    for layers in base_model.layers:
        layers.trainable = False # Freeze all base model layers before training
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=50, batch_size=32, validation_data=val_data, callbacks=[tensor_board, monitor, lr])
    model.save('Mask_detector_model1') # model will be saved as a keras model which can be loaded using load model function in keras


path = r'I:\AI\tf _2021\Mask_Detection\source\dataset'
model_config, base_model_config = model()
train_data, val_data = data_generator(path)
train(model_config, base_model_config, train_data, val_data)

