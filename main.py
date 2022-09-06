from src import App


if __name__ == '__main__':
    app = App()
    app.rnn_video_emotion_recognition(0)  # For webcam emotion recognition
    # app.rnn_video_emotion_recognition('data/videos/vid1.mp4')  # For emotion recognition from file
    # app.cnn_predict_image_file('data/img/00001.jpg')
