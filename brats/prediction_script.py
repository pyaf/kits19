from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
import tables
import numpy as np
import time
from unet3d.prediction import prediction_to_image

def prediction_script(data_file):
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    custom_objects["InstanceNormalization"] = InstanceNormalization

    model = load_model('isensee_2017_model.h5', custom_objects=custom_objects)
    data_file = tables.open_file('brats_data.h5', "r")
    print(data_file.root.subject_ids[0])
    print(np.shape(data_file.root.data[0]))

    # predict:
    start = time.time()
    affine = data_file.root.affine[0]
    prediction = model.predict(np.asarray([data_file.root.data[0]]))
    prediction_image = prediction_to_image(prediction, affine)
    prediction_image.to_filename("test.nii.gz")
    print(time.time() - start)
    data_file.close()

# if __name__ == '__main__':
prediction_script('case_00000.nii.gz')