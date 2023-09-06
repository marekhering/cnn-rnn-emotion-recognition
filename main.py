from src import App


if __name__ == '__main__':
    app = App()
    # app.videos_inference()  # For getting inference results from videos files
    app.video_inference(0, vis=True, save=False)  # For webcam emotion recognition
    # path = "data/input/videos/1dpri38e-jpub-kkut-8r0k-drdpmvehaaub_GetHello.mp4"
    # app.video_inference(path, vis=False, save=False)  # For emotion recognition from file
