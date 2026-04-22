# Where is Waldo CNN Detector
A program with a Convolutional Neural Network to be able to find Waldo from the book series 'Where is Waldo?' The model works by chopping the book page into hundreds of small 128×128 pixel windows and feeding each one through a CNN, which outputs a score between 0 and 1 representing how likely Waldo is to be in that patch. Once every tile has been scored, the one with the highest score is highlighted as the predicted location. If that's wrong, the user can step through the next-best matches one by one.

Training data is generated on-the-fly rather than from a fixed image set. For each positive sample, a 128×128 crop is placed around Waldo's bounding box with a small random jitter of up to ±16 pixels, and a visibility score is calculated based on how much of Waldo actually falls within the crop window. Negative samples are drawn from random locations guaranteed not to overlap with Waldo. Random brightness, contrast, and Gaussian noise augmentations are applied to both, and class balance is enforced dynamically since non-Waldo tiles vastly outnumber Waldo ones. This approach means the model sees a virtually unlimited number of unique tile variations across a typical training run of around 19 million tile exposures.

The CNN itself was trained to recognise Waldo's distinctive local features — red/white stripes, glasses, and a cane — from thousands of cropped training examples. The key limitation is that each tile is judged in complete isolation, so the model has no awareness of the surrounding scene. This means it can struggle to tell Waldo apart from other similarly dressed characters, since it can't use any broader context to help make that distinction.

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

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE) - see the [LICENSE](LICENSE) file for details.

