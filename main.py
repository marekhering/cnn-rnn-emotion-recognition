from src import App


if __name__ == '__main__':
    app = App()
    app.videos_inference()  # For getting inference results from videos files
    # app.video_inference(0, vis=True)  # For webcam emotion recognition
    # app.video_inference('data/input/videos/vid1.mp4', vis=True)  # For emotion recognition from file
    # app.image_inference('data/input/img/00001.jpg')
