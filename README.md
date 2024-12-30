# Message-Spam-Detection
This project implements a message spam detection system using a machine learning model. The model is trained on a dataset of text messages and uses a Random Forest Classifier to predict whether a given message is "Spam" or "Not Spam". The system has both a command-line interface and a graphical user interface (GUI) for ease of use.

## Features
- **Spam Detection**: Classifies a message as either "Spam" or "Not Spam".
- **Data Cleaning & Preprocessing**: Handles punctuation, digits, stopwords, and lemmatization of input text.
- **Machine Learning Model**: Utilizes a Random Forest Classifier trained on a cleaned dataset.
- **GUI Interface**: A simple GUI built using `tkinter` for user-friendly input and output display.

## Requirements
To run this project, you'll need the following Python libraries:

- `pandas`
- `scikit-learn`
- `nltk`
- `tkinter` (comes pre-installed with Python)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/farhat-1203/Message-Spam-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Message-Spam-Detection
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

```
Message-Spam-Detection
├── Cleaned_Dataset.csv          # The cleaned dataset used for model training
├── message_detection.py         # Main Python file for spam detection
├── README.md                    # Project documentation
├── requirements.txt             # List of dependencies for the project
└── .gitignore                   # Git ignore file (if using Git)
```
