# Where is Waldo CNN Detector
A program with a Convolutional Neural Network to be able to find Waldo from the book series 'Where is Waldo?'

There are about 36 Keras models available in the `Keras Models` folder, from model generation 10 the accuracy improved drastically however it is still not reccommended to use these models in any commercial or professional capacity.

---


# Getting started

### Prerequisites
-   Python 3.12+
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Jetss3/Where-is-Waldo-CNN-Detector.git
    cd Where-is-Waldo-CNN-Detector
    ```

2.  **Create a virtual environment** (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Use a pre-trained model**
    There are multiple pre-trained model to use, reccommended is to use the `Keras Models/prototype11k.keras` model provided in the keras models folder.

### Usage
Run the GUI using the following command:
    ```bash
    python main.py
    ```
