from src import App


if __name__ == '__main__':
    app = App()
    app.visualize_inference(0)  # For webcam emotion recognition
    # app.visualize_inference('data/videos/vid1.mp4')  # For emotion recognition from file
    # app.inference_image('data/img/00001.jpg')
