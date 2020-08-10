# Face-Recognition
Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf).

Face recognition problems commonly fall into two categories:
1. **Face Verification** : "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
2. **Face Recognition**  :  "who is this person?". For example, employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.


FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

## Naive Face Verification
In Face Verification, you're given two images and you have to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person!

<p align = 'center'>
  <img src = '/images/pixel_comparison.png'>
</p>
- Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on.
- You'll see that rather than using the raw image, we can learn an encoding, f(img) .
- By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

## Encoding face images into a 128-dimensional vector
### Using a ConvNet to compute encodings

The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning, let's load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy et al.](https://arxiv.org/abs/1409.4842).
It is implemented in the file inception_blocks_v2.py

The key things are: 
1. This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of  m  face images) as a tensor of shape  (m,n<sub>C<sub>,n<sub>H</sub>,n<sub>W</sub>)=(m,3,96,96).
2. It outputs a matrix of shape  (m,128)  that encodes each input face image into a 128-dimensional vector.

By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings to compare two face images as follows:

<p align = 'center'>
  <img src = '/images/distance_kiank.png'>
</p>

By computing the distance between two encodings and thresholding, you can determine if the two pictures represent the same person.

So, an encoding is a good one if:
- The encodings of two images of the same person are quite similar to each other.
- The encodings of two images of different persons are very different.

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart.
<p align = 'center'>
  <img src = '/images/triplet_comparison.png'>
</p>

In the next part, we will call the pictures from left to right: Anchor (A), Positive (P), Negative (N)

## The Triplet Loss
For an image  x , we denote its encoding  f(x) , where  f  is the function computed by the neural network.
<p align = 'center'>
  <img src = '/images/f_x.png'>
</p>

Training will use triplets of images  (A,P,N) :
- A is an "Anchor" image--a picture of a person.
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from our training dataset. We will write  (A<sup>(i)</sup>,P<sup>(i)</sup>,N<sup>(i)</sup>) to denote the i-th training example.

We'd like to make sure that an image A<sup>(i)</sup> of an individual is closer to the Positive P<sup>(i)</sup> than to the Negative image N<sup>(i)</sup> by at least a margin α:
<p align = 'center'>
  <img src = 'Screenshot (139).png'>
</p>

We would thus like to minimize the following "triplet cost":
<p align = 'center'>
  <img src = 'Screenshot (138).png'>
</p>

## References
- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015).[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014).[DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
- The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- This implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet

