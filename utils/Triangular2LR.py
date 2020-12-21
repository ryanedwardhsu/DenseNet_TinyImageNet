import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class TriangularCyclicalLR(keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, stepSize):
        """
        Inputs:
        :min_lr: minimum learning rate
        :max_lar: maximum learning rate
        :step_size: number of iterations to half-cycle
        """
        super(TriangularCyclicalLR, self).__init__() # <- Calls Keras callback init
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_diff = max_lr - min_lr
        self.stepSize = stepSize
        self.switch = 1 # Switch to change increasing or decreasing lr. Initially 1 for positive gradient
        self.curr_steps = 0 # Steps to keep track of our iteration number
        # print("Updated step size: ", self.stepSize)Simial

    def on_epoch_begin(self, epoch, logs=None):
        # Detect start of a new epoch and set re-set the base LR accordingly.
        # This allows us to not have to re-compile and manually set lr at the start.
        print("Current epoch is: ", epoch)
        if epoch == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_begin(self, batch, logs=None):
        # Tensorflow on_batch_end will be our first entry point to modify the LR
        # We can use keras.backend to get the model's current LR, modify it accordinly and plug back into model
        # Source: https://www.tensorflow.org/guide/keras/custom_callback
        
        # Fetch current lr from the optimizer
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # If we've reached the 'stepSize' threshold, we reset curr steps and change direction of LR
        # self.switch allows us to modify the gradient of LR change.
        # self.curr_steps keeps track of where we are w.r.t the designated stepSize
        # print("Checking step size is: ", self.stepSize)
        # print("Lr for batch ", batch, " is: ", lr)
        
        if self.curr_steps >= self.stepSize:
            self.switch = self.switch * -1
            # If switch is now positive again, cycle has ended
            # Per Triangular2 LR, we cut the LR difference by half
            if self.switch == 1:
                self.lr_diff = self.lr_diff/2
            self.curr_steps = 0
        # Modify learning rate according to: https://arxiv.org/pdf/1506.01186.pdf
        # The amount of modification can be determined by taking diff between max and min lr 
        # and dividing by steps. We then multiply by switch, which is either 1 or -1 depending on step.
        lr += (self.lr_diff/self.stepSize) * self.switch
        # print("New lr is: ", lr)

        # We can then re-plug the new lr back into the model via the backend.set_value function:
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        # At the end, increment the current steps
        self.curr_steps += 1
