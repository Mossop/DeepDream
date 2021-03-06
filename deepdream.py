import numpy as np
import scipy.ndimage as nd
import os, sys
from tempfile import mkstemp
import PIL.Image
from google.protobuf import text_format

model = 'bvlc_googlenet'
layer = 'inception_4c/output'

root = os.path.dirname(sys.argv[0])

caffedir = os.path.join(root, 'caffe')
sys.path.append(os.path.join(caffedir, 'python'))

import caffe

def saveimage(a, path, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    name = path + '.' + fmt
    PIL.Image.fromarray(a).save(name, fmt)
    print name

def loadmodel(model=model):
    model_path = os.path.join(caffedir, 'models', model) # substitute your path here
    net_fn   = os.path.join(model_path, 'deploy.prototxt')

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    return model

def loadnet(model=model):
    model_path = os.path.join(caffedir, 'models', model) # substitute your path here
    param_fn = os.path.join(model_path, model + '.caffemodel')

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = loadmodel(model)
    model.force_backward = True

    (oshandle, path) = mkstemp()
    os.fdopen(oshandle, 'w').write(str(model))

    try:
        return caffe.Classifier(path, param_fn,
                                mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                                channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    finally:
        os.unlink(path)

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, end, step_size=1.5, jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, end, iter_n=10, octave_n=4, octave_scale=1.4, clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end, clip=clip, **step_params)

            # display step
            #vis = deprocess(net, src.data[0])
            #if not clip: # adjust image contrast if clipping is disabled
            #    vis = vis*(255.0/np.percentile(vis, 99.98))
            #ename = '-'.join(end.split('/'))
            #saveimage(vis, '{}-{}-{}'.format(octave, i))
            #print octave, i, end, vis.shape

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Computer dreams')
    parser.add_argument('image', metavar='IMAGE',
                        help='the image to dream about')
    parser.add_argument('--model', dest='model', default='bvlc_googlenet',
                        help='the model to use')
    parser.add_argument('--output', dest='output', default=None,
                        help='the filename to save the result as')
    parser.add_argument('--layer', dest='layer', default=layer,
                        help='the layer to optimise')
    args = parser.parse_args()

    img = np.float32(PIL.Image.open(args.image))
    output = args.output
    if output is None:
        (name, extract) = os.path.splitext(args.image)
        output = name + '_' + '-'.join(args.layer.split('/'))

    net = loadnet(args.model)
    frame = deepdream(net, img, args.layer)

    saveimage(frame, output)
