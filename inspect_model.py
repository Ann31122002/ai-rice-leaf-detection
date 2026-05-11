from tensorflow import keras
import sys

try:
    print('Loading ResNet model...')
    model = keras.models.load_model('models/keras_resnet_finetuned.h5')
    print('Model loaded successfully')
    print('Input shape:', model.input_shape)
    print('Output shape:', model.output_shape)
    print('Layers:')
    for i, layer in enumerate(model.layers):
        print(f'{i}: {layer.name} - {layer.__class__.__name__}')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)