#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines for training the network.

"""


__all__ = (
    'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model

def data_to_vector(p, sign):
    vec = numpy.zeros(1 + len(model.CLASSES), dtype='float32')
    vec[0] = 1 if p else 0
    vec[model.CLASSES.index(sign) + 1] = 1

    return vec

def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        sign = fname.split("/")[1][9:len(fname) - 7 - len(fname.split("/")[0])]
        p = fname.split("/")[1][len(fname) - 6 - len(fname.split("/")[0])] == '1'
        yield im, data_to_vector(p, sign)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped
        

@mpgen
def read_batches(batch_size):
    g = gen.generate_ims()
    def gen_vecs():
        for im, s, p in itertools.islice(g, batch_size):
            yield im, data_to_vector(p, s)

    while True:
        yield unzip(gen_vecs())


def get_loss(y, y_):
    # Calculate the loss from predicted sign being incorrect.  Don't count loss from
    # non-present sign.
    #y = tf.Print(y, [y], summarize=100)
    class_loss = tf.nn.softmax_cross_entropy_with_logits(y[:, 1:], y_[:, 1:])
    class_loss *= tf.reshape([y_[:, 0], y_[:, 0]], [-1, 1])
    class_loss = tf.reduce_sum(class_loss)

    # Calculate the loss from presence indicator being wrong.
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(y[:, 0], y_[:, 0])
    #presence_loss = tf.Print(presence_loss, [class_loss, presence_loss], summarize=100)
    presence_loss = tf.reduce_sum(presence_loss)

    return class_loss, presence_loss, class_loss + presence_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    x, y, params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, 1 + len(model.CLASSES)])

    class_loss, presence_loss, loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    best = tf.argmax(y[:, 1:], 1)
    correct = tf.argmax(y_[:, 1:], 1)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.global_variables_initializer()

    def do_report():
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      class_loss,
                      presence_loss,
                      loss],
                      feed_dict={x: test_xs, y_: test_ys})

        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(model.CLASSES[c], pc,
                                           model.CLASSES[b], float(pb))

        print ("B{:3d} loss: {} ").format(
            batch_idx,
            r[6])

    def do_batch():
        #writer = tf.summary.FileWriter("./log", sess.graph)
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(learn_rate=0.001,
          report_steps=5,
          batch_size=50,
          initial_weights=initial_weights)

