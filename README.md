# Emotion-age-and-ethnicity-Estimation

1. Data Preprocess
Cropping of faces from images
Face hallucination for Low Resolution Cropped Image
Image Augmentation

2. Model Architecture
Depth wise Separable Convolutional Neural Networks

The model consists of 4 CNN blocks. Each i-th block is a set of 2 depthwise separable conv layers.
         Each layer in the i-th block has kernel size given by the i-th element from `kernels_size` parameter,
         and a number of output channels given by the i-th element from `n_filters` parameter.

        :param dropout: float between 0 and 1 for the dropout rate
        :param n_class: number of classes <==> output shape
        :param n_channels: input image number of channels, e.g. 3 for RGB and 1 for grayscale
        :param n_filters: list of ints, each i-th element is the number of output channels for the conv layers
                of i-th conv block
        :param kernels_size: list of ints, each i-th element is the kernel size for the conv layers
                of i-th conv block

Code snippet for multimodel
```
        self.output_age = nn.Linear(128, self.n_class[0])
        self.output_emotion = nn.Linear(128, self.n_class[1])
        self.output_ethnicity = nn.Linear(128, self.n_class[2])
```

Code snippet for pytorch neural network forward method
```
def forward(self, x):
        x = self.conv_base(x)
        age = self.output_age(x)
        emotion = self.output_emotion(x)
        ethnicity = self.output_ethnicity(x)

        return age, emotion, ethnicity
```

3. Training

4. Evaluation

5. Testing

6. Conclusion
