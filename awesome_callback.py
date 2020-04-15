import os

import tensorflow.keras.backend as tf_core
from tensorflow.keras.callbacks import Callback
from tensorflow.summary import create_file_writer, scalar


class MyCallback(Callback):
    def __init__(self, run_name, save_every, base_dir="experiments"):
        super().__init__()
        self.save_counter = 0
        self.least_loss = -1
        os.makedirs(base_dir)
        self.base_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "logs"), exist_ok=True)
        self.summary_writer = create_file_writer(os.path.join(self.base_dir, "logs"))
        self.summary_writer.set_as_default()
        self.iters_since_last_model_save = save_every + 1
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs=None):
        iter_no = tf_core.get_value(self.model.optimizer.iterations)
        train_loss = logs["loss"]
        val_loss = logs["val_loss"]
        scalar("train_loss", data=train_loss, step=iter_no)
        scalar("val_loss", data=val_loss, step=iter_no)
        self.iters_since_last_model_save += 1
        if self.least_loss < 0 or train_loss < self.least_loss:
            self.least_loss = train_loss
            print("\nLoss decreased in iter {}".format(iter_no))
            if self.iters_since_last_model_save > self.save_every:
                print(
                    "\nSaving model at iteration {} with loss {}".format(
                        iter_no, train_loss
                    )
                )
                # save_model(self.model, os.path.join(self.base_dir, "models", "model-{0:.4f}.h5".format(train_loss)))
                self.iters_since_last_model_save = 0

    def apply_lr(self):
        tf_core.set_value(self.model.optimizer.lr, tf_core.get_value(0.0001))
