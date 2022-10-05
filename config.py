class GeneralConfig:
    DEFAULT_QUIT_BUTTON = 'q'
    MAX_BUFFER_SIZE = 150


class PathConfig:
    CNN_MODEL_PATH = 'data/models/simple_cnn_ccc_loss.0001-0.7792.hdf5'
    RNN_MODEL_PATH = 'data/models/gru_ccc_loss-epoch_17-loss_0.6913-ccc_v_0.2252-ccc_a_0.3922.hdf5'
    VALENCE_AROUSAL_SPACE_PATH = 'data/utils/valence_arousal_space.jpg'
    FACE_RECOGNITION_MODEL_PATH = 'data/models/haarcascade_frontalface_default.xml'
    VIDEOS_PATH = 'data/input/videos'
    OUTPUT_VIDEOS = 'data/output/videos'


class FrameConfig:
    MAIN_FRAME_SIZE = (800, 1200)
    FRAME_BACKGROUND = (255, 255, 255)


class AnalysisConing:
    DELAY = 20  # seconds
    SENSITIVITY = 1.5  # Value refers to the multiplier for the standard deviation
