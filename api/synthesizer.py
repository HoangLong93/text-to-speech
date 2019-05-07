# import io
# import numpy as np
# import tensorflow as tf
# from hparams import hparams
# from librosa import effects
# from models import create_model
# from text import text_to_sequence
# from util import audio


# class Synthesizer:
#   def load(self, checkpoint_path, model_name='tacotron'):
#     print('Constructing model: %s' % model_name)
#     inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
#     input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
#     with tf.variable_scope('model') as scope:
#       self.model = create_model(model_name, hparams)
#       self.model.initialize(inputs, input_lengths)
#       self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

#     print('Loading checkpoint: %s' % checkpoint_path)
#     self.session = tf.Session()
#     self.session.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(self.session, checkpoint_path)


#   def synthesize(self, text):
#     cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
#     seq = text_to_sequence(text, cleaner_names)
#     feed_dict = {
#       self.model.inputs: [np.asarray(seq, dtype=np.int32)],
#       self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
#     }
#     wav = self.session.run(self.wav_output, feed_dict=feed_dict)
#     wav = audio.inv_preemphasis(wav)
#     wav = wav[:audio.find_endpoint(wav)]
#     out = io.BytesIO()
#     audio.save_wav(wav, out)
#     return out.getvalue()

# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav
from scipy.io.wavfile import write
import os
import numpy as np


def synthesize(reqSent):
    if not os.path.exists(hp.sampledir): os.mkdir(hp.sampledir)

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    # Load data
    texts = load_data("synthesize", reqSent)
    print(texts)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./')); print("Restored!")

        # Feed Forward
        ## mel
        y_hat = np.zeros((texts.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
        for j in tqdm.tqdm(range(200)):
            _y_hat = sess.run(g.y_hat, {g.x: texts, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]
        ## mag
        mags = sess.run(g.z_hat, {g.y_hat: y_hat})
        for i, mag in enumerate(mags):
            print("File {}.wav is being generated ...".format(i+1))
            audio = spectrogram2wav(mag)
            write(os.path.join(hp.sampledir, '{}.wav'.format(i+1)), hp.sr, audio)

if __name__ == '__main__':
    synthesize()
    print("Done")