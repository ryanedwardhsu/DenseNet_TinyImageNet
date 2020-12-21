import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from sklearn.utils import shuffle as sfl

def spaceToDepth(input_tensor):
    # After MaxPool, 1st block has shape (32,32), 2nd block shape (16,16), 3rd block shape (8x8)
    # Hence to ensure correct input to Concatenate, need to 'split' previous block in 2.
    # Hence, set block_size = 2
    split_tensor = tf.nn.space_to_depth(input_tensor, block_size=2)
    return split_tensor

# def imgAug(input_img):
def translate(input_img, shift_height, shift_width):
  """
  Translate self.x by the values given in shift.
  :param shift_height: the number of pixels to shift along height direction. Can be negative.
  :param shift_width: the number of pixels to shift along width direction. Can be negative.
  """
  translated = np.roll(input_img, (shift_width, shift_height), axis=(0, 1))
  print('Current translation: ', shift_height, shift_width)
  return translated

def rotate_img(input_img, angle=0.0):
  """
  Rotate self.x by the angles (in degree) given.
  :param angle: Rotation angle in degrees.
  :return rotated: rotated dataset
  - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
  """
  
  rotated = rotate(input_img, angle,reshape=False,axes=(0, 1))
  print('Currrent rotation: ', angle)
  return rotated

def h_flip(input_img):
  """
  Flip self.x according to the mode specified
  :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
  :return flipped: flipped dataset
  """
  flipped = np.fliplr(input_img)
  return flipped

def v_flip(input_img):
  """
  Flip self.x according to the mode specified
  :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
  :return flipped: flipped dataset
  """
  flipped = np.flipud(input_img)
  return flipped

def add_noise(input_img, amplitude):
  """
  Add random integer noise to self.x.
  :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                  then 1000 samples will be noise-injected.
  :param amplitude: An integer scaling factor of the noise.
  :return added: dataset with noise added
  """
  added_noise = input_img + np.random.normal(0, 0.07, input_img.shape)*amplitude
  added_noise = np.clip(added_noise, 0, 255)
  return added_noise