# First Experimental Model! 
Needs a lot of work but it is something!! :D

![image](https://github.com/Andrew-Pynch/speedchallenge/blob/master/models/demo.gif?raw=true)

# TODO

<!-- prettier-ignore -->
| STATUS | FEATURE   | DESCRIPTION  |
|---|-----------|--------------|
| 🎉 | Resnet 151 Model | Basically train a model to directly map from single frame samples to speed predictions. Unfortunately This did not work very well. |
| ❗ | Label Pipeline | Image Transforms Pipeline: mp4 --> images |
| ❗ | Cleanup utils | Utils has a bunch of useful functions for this project, needs some cleanup |
| ❗ | . | . |
| ❗ | . | . |

## Initial Experiment Results

First model was basically a super nieve attempt to map directly from single frames to speed predictions. Used a Resnet151 pretrained on imagene to directly try and predict the speed of a given frame. Model appeared to do well in training / validation but WAY overfit in the testing process. It basically constantly outputting the average speed of the training set (which is expected but unfortunate :/). Since it is probably not possible (or at the very least easy) to predict the speed of a car from a single frame from a video, I going to explore options that make use of the temporal data. 3d convolutions? :thinking:

I am also going to try and fit this data using an LSTM.
