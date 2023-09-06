from pathlib import Path


class GeneralConfig:
    DEFAULT_QUIT_BUTTON = 'q'
    MAX_BUFFER_SIZE = 600
    DEFAULT_FPS = 30
    REQUIRED_FPS = 20


OUTPUT_PATH = "data/output"


class PathConfig:
    CNN_MODEL_PATH = 'data/models/simple_cnn_ccc_loss.0001-0.7792.hdf5'
    RNN_MODEL_PATH = 'data/models/gru_ccc_loss-epoch_17-loss_0.6913-ccc_v_0.2252-ccc_a_0.3922.hdf5'
    VALENCE_AROUSAL_SPACE_PATH = 'data/utils/valence_arousal_space.jpg'
    FACE_RECOGNITION_MODEL_PATH = 'data/models/haarcascade_frontalface_default.xml'
    EVAL_VIDEOS_PATH = 'data/eval/videos'
    VIDEOS_LINKS_PATH = 'data/utils/links.txt'
    VIDEOS_PATH = 'data/input/videos'
    OUTPUT_VIDEOS_PATH = f'{OUTPUT_PATH}/videos'
    OUTPUT_VA_PATH = f'{OUTPUT_PATH}/va'
    PLOTS_PATH = 'data/plots'
    CM_PATH = 'data/confusion_matrices'

    @staticmethod
    def mkdir(_path: str):
        Path(_path).mkdir(parents=True, exist_ok=True)


class FrameConfig:
    # Additional border in found face frame
    FACE_FRAME_OFFSET = 0

    MAIN_FRAME_SIZE = (900, 1500)
    FRAME_BACKGROUND = (255, 255, 255)


class AnalysisConing:
    DELAY = 10  # seconds
    CNN_STD_SENSITIVITY = 2.5  # Value refers to the multiplier for the standard deviation
    RNN_STD_SENSITIVITY = 1.5    # Value refers to the multiplier for the standard deviation

    CNN_MOVING_AVERAGE_WINDOW = 8
    CNN_DERIVATIVE_MOVING_AVERAGE_WINDOW = 1
    RNN_MOVING_AVERAGE_WINDOW = 16
    RNN_DERIVATIVE_MOVING_AVERAGE_WINDOW = 2
