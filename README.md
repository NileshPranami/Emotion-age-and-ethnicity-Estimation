# Emotion-age-and-ethnicity-Estimation

1. Data Preprocess
Cropping of faces from images
Face hallucination for Low Resolution Cropped Image
Image Augmentation

2. Model Architecture
Depth wise Separable Convolutional Neural Networks
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
