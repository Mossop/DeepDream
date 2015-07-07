import numpy as np
import scipy.ndimage as nd
import PIL.Image
import os

from deepdream import deepdream, loadnet, saveimage

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Computer dreams')
    parser.add_argument('image', metavar='IMAGE',
                        help='the image to dream about')
    parser.add_argument('--model', dest='model', default='bvlc_googlenet',
                        help='the model to use')
    parser.add_argument('--output', dest='output', default='output',
                        help='the directory to output results to')
    args = parser.parse_args()

    image = np.float32(PIL.Image.open(args.image))
    (name, ext) = os.path.splitext(os.path.basename(args.image))

    net = loadnet(args.model)

    for layer in net._layer_names:
        try:
            print("Testing layer %s" % layer)
            frame = deepdream(net, image, layer)
            saveimage(frame, os.path.join(args.output, name + '_' + '-'.join(layer.split('/'))))
        except:
            print("Invalid layer name %s" % layer)
