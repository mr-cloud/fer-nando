## anatomy of this project
### dataflow
1. 48 x 48 gray images
2. features extraction -> landmarks + HOG + image (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
3. modeling with checkpoint
4. predicting with static image rescaling into the proper size before entering into model.
5. video streaming captures frames, detecting faces, extracting features and predicting as a static image.



## benchmark
### training with default parameters
training time = 13688.0 sec
saving model...
evaluating...
  - validation accuracy = 54.3
loading dataset Fer2013...
building model...
start evaluation...
loading pretrained model...
--
Validation samples: 3589
Test samples: 3589
--
evaluating...
  - validation accuracy = 54.3
  - test accuracy = 54.8
  - evalution time = 58.9 sec
