# DeepDream
Experiments with [Google's DeepDream code](http://googleresearch.blogspot.ch/2015/07/deepdream-code-example-for-visualizing.html). Code originally taken from https://registry.hub.docker.com/u/mjibson/deepdream/

## deepdream.py
Runs a single pass on the passed image. Run `deepdream.py -h` for command line options.

## redream.py
Iteratively feeds the generated image back into itself zooming in a little at a time. Run `redream.py -h` for command line options.

## testdream.py
Iterates through all the layers in the model and attempts to produce an image for each one. Some layers don't work and often this will end with a crash but it's a quick way to get an idea of what each layer does. Run `testlayers.py` for command line options.