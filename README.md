## Face Age Classification Using Fine-tuning of Pretrained VGGFace Model

This project provides a web-based interface to classify a person's age category based on a given image. The available categories are:

- Children (`anak-anak`)
- Adults (`dewasa`)
- Teens (`remaja`)

The model is built using a fine-tuned Pretrained VGGFace model. The app interface is built using Streamlit.

### How to Use

1. Visit the deployed web application.
2. Upload an image containing a face using the "Upload an image" button.
3. The uploaded image will be displayed.
4. The application will automatically detect the face and display the cropped face image.
5. Click on the "Predict" button to classify the face's age category.
6. The result will be displayed prominently on the screen.

### Key Features

- Uses a Pretrained VGGFace model fine-tuned for face age classification.
- Efficient face detection using MTCNN.
- Interactive and user-friendly interface using Streamlit.
- Efficient handling and caching of a model using Streamlit cache.
- Can handle images in `jpg`, `jpeg`, and `png` formats.

### Installation & Running Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2. Install the required packages:
    ```bash
    pip install streamlit tensorflow mtcnn opencv-python-headless
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any improvements or fixes.

### Credits

- Face Detection: [MTCNN](https://github.com/ipazc/mtcnn)
- Model Fine-tuning: Pretrained VGGFace model
- Web Interface: [Streamlit](https://www.streamlit.io/)
- Custom metric `f1_score` is used during model training and evaluation.