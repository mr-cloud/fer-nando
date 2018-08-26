"""
@AmineHorseman
Sep, 1st, 2016
"""
import os

class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    shape_predictor_path='shape_predictor_68_face_landmarks.dat'
    trunc_trainset_to = -1  # put the number of train images to use (-1 = all images of the train set)
    trunc_validationset_to = -1
    trunc_testset_to = -1

class Network:
    input_size = 48
    # number of expressions should match the number of labels in dataset preprocessing
    output_size = 5
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
    use_hog_and_landmarks = True
    use_hog_sliding_window_and_landmarks = True
    use_batchnorm_after_conv_layers = True
    use_batchnorm_after_fully_connected_layers = False

class Hyperparams:
    keep_prob = 0.89383
    learning_rate = 0.09632
    learning_rate_decay = 0.6972
    decay_step = 50
    optimizer = 'momentum'  # {'momentum', 'adam'}
    optimizer_param = 0.75463   # momentum value for Momentum optimizer, or beta1 value for Adam
    # FIXME initialization method selection


class Training:
    batch_size = 128
    epochs = 15
    snapshot_step = 10000
    vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/chk"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0 # in hours
    save_model = True
    save_model_path = "best_model/saved_model.bin"

class VideoPredictor:
    # Number of emotions should be the same as NETWORK.output_size
    # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = False
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = False
    frame_rate = 0.01
    time_to_wait_between_predictions = frame_rate
    time_to_wait_between_display = 0.03
    title_change_interval = 1.0


def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
VIDEO_PREDICTOR = VideoPredictor()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)
