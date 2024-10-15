# Telugu Text Summarization

This project focuses on developing a text summarization tool specifically for the Telugu language. Utilizing state-of-the-art Natural Language Processing (NLP) techniques, it aims to extract key information and generate concise summaries from larger Telugu texts.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Telugu Text Summarization tool employs advanced models, such as mT5 and mBART, to condense Telugu texts into shorter, coherent summaries. The project aims to improve accessibility and understanding of Telugu literature and documents by providing succinct overviews.

## Features

- **Language Support**: Specifically designed for the Telugu language.
- **Model Utilization**: Leverages pre-trained models like mT5 and mBART for effective summarization.
- **Custom Dataset**: Fine-tunes models on a custom dataset (`telugu_ilsum_2024_train.csv`) containing approximately 1800 rows.
- **User-Friendly Interface**: Easy-to-use interface for inputting text and obtaining summaries.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**: Download the project from the repository.
   ```bash
   git clone https://github.com/your_username/telugu-text-summarization.git
2. **Navigate to the Project Directory**: Change your working directory to the project folder.
    cd telugu-text-summarization

3. **Install Required Dependencies**: Ensure you have the necessary Python packages installed:
    pip install -r requirements.txt

## Usage
Once the dependencies are installed and the project is set up, follow these steps to use the summarization tool:
- Load the Model: Open your notebook or Python script and load the pre-trained model.

## Technologies Used
- Python: Core programming language used for the project.
- mT5: A pre-trained model used for text summarization.
- mBART: Another pre-trained model utilized for effective summarization tasks.
- Pandas: For data manipulation and handling the dataset.
- TensorFlow/PyTorch: Depending on the model implementation, either framework is used for model training and inference.
