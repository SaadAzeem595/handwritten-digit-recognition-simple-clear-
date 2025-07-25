Handwritten Digit Recognition Web App
An interactive web application built with TensorFlow and Streamlit that recognizes handwritten digits (0–9) using a deep learning model trained on the MNIST dataset.

📌 Overview
This project demonstrates how to integrate a deep learning model into a web application for real-time digit recognition. Users can upload an image of a handwritten digit, and the app predicts the digit with confidence scores.

🚀 Features
Deep Learning Model: Achieves 99% test accuracy on the MNIST dataset.

Image Upload: Users can upload handwritten digit images (PNG/JPG).

Preprocessing Pipeline:

Grayscale conversion

Background inversion

Cropping & centering

Resizing to 28×28 (MNIST format)

Normalization

Confidence Scores: Displays model prediction probabilities for all digits (0–9).

Streamlit UI: Interactive and easy-to-use interface.

🛠 Tech Stack
Python

TensorFlow/Keras

OpenCV

PIL (Pillow)

NumPy

Streamlit

📂 Project Structure
bash
Copy
Edit
.
├── app.py               # Main Streamlit app
├── model.h5             # Pre-trained MNIST model
├── requirements.txt     # Python dependencies
├── mnist_images/        # Sample images (optional)
└── README.md            # Project documentation
⚙️ Installation & Setup
Clone the Repository

bash
Copy
Edit
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Create Virtual Environment (Optional but Recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit App

bash
Copy
Edit
streamlit run app.py
📊 Model Performance
Training Accuracy: 99.58%

Test Accuracy: 99.04%

🖼 Demo

(Add a real screenshot or GIF of your app running here)

📅 Future Enhancements
Add webcam integration for real-time digit capture.

Add drawing canvas for custom digit input.

Deploy on Streamlit Cloud or Heroku for public access.

🤝 Contributing
Contributions are welcome! Feel free to fork this repo and create a pull request.

📜 License
This project is licensed under the MIT License.