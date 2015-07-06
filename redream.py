import numpy as np
import scipy.ndimage as nd
import PIL.Image
import os

from deepdream import deepdream, loadnet, saveimage

layer = 'inception_4c/output'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Computer dreams')
    parser.add_argument('image', metavar='IMAGE',
                        help='the image to dream about')
    parser.add_argument('--model', dest='model', default='bvlc_googlenet',
                        help='the model to use')
    parser.add_argument('--layer', dest='layer', default=layer,
                        help='the layer to optimise')
    parser.add_argument('--scale', dest='scale', type=float, default=0.05,
                        help='the scale coefficient')
    parser.add_argument('--iterations', dest='iterations', type=int, default=100,
                        help='the number of iterations to run')
    parser.add_argument('--output', dest='output', default='output',
                        help='the directory to output frames to')
    args = parser.parse_args()

    frame = np.float32(PIL.Image.open(args.image))

    net = loadnet(args.model)

    h, w = frame.shape[:2]
    s = args.scale
    for i in xrange(args.iterations):
        frame = deepdream(net, frame, args.layer)
        saveimage(frame, os.path.join(args.output, "frame-%04d" % i))
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
