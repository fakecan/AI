from keras.models import load_model
import Convert_Keras_Coreml, Convert_Keras_Coreml2
model = load_model('./keras_model.h5')
class_labels = ['cheomseongdae', 'colosseum', 'damyang_metasequoia', 'n_seoul_tower', 'pyramid']
coreml_model = Convert_Keras_Coreml(model, input_names='image', image_input_names='image',
                                    class_labels=class_labels, is_bgr=True)
coreml_model.save('./convert_coreml_keras_model.mlmodel')
