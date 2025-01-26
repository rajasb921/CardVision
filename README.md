# **CardVision: AI-Powered Card Classification and Poker Simulation**

## About the Project

This project combines computer vision, machine learning, and interactive gaming to create a unique card classification and poker simulation tool. 
It leverages a pre-trained Convolutional Neural Network (CNN) model to analyze and classify playing cards, enabling two primary functionalities:

1. **Card Classification**:  

   The project accurately classifies playing cards (e.g., suits and ranks) using image inputs.
   This capability is demonstrated in the **main classification program** (`main.py`), which allows users to test the model with various card sets.

2. **Poker Simulation**:  
   The interactive poker simulation (`poker.py`) offers a dynamic gameplay experience.
   It integrates real-time card classification with statistical odds calculation to simulate a Texas Hold'em poker game.
   Players can make strategic decisions based on data-driven recommendations (e.g., bet, check, fold) while competing against a simulated opponent.

---

#### Key Features

- **Pre-Trained Model**: Utilizes a CNN for high-accuracy card recognition. The training code for the model can be found in the CV_Final Jupyter Notebook
- **Interactive Gameplay**: Simulates a poker game with real-time card classification and odds calculation.
- **Custom Odds Calculation**: Dynamically evaluates pre-flop win probabilities against an opponent.
- **User-Friendly Design**: Easy-to-use interface for both classification and gameplay modes.

## Download and Verify Imports

To set up and verify all the necessary imports for this project, follow these steps:

---

#### 1. Install Required Libraries

The project requires several Python libraries. You can install them using `pip`. Run the following command:

```bash
pip install opencv-python numpy tensorflow scikit-learn matplotlib seaborn
```

---

#### 2. Install the Custom Library: `poker_calc`

#### *(Note: This part is not necessary for the main demonstration in the project)*

The interactive part of this project relies on the `poker_calc` library. The required code is provided as a `.tar.gz` file. You will need to install it manually:

1. **Download the File**  
   Ensure you have the `poker_calc-0.0.2.tar.gz` file. Move it to a location on your system, for example, `/path/to/your/downloads/`.

2. **Install the Library**  
   Use the following command, replacing `/path/to/your/downloads/` with the actual location of the file:

   ```bash
   pip install "/path/to/your/downloads/poker_calc-0.0.2.tar.gz"
   ```

---

#### 3. Verify Downloads and Imports

To ensure all libraries have been installed correctly, run the following bash commands:

1. **Verify Installed Libraries**
   Run this command to check that the necessary libraries are installed:

   ```bash
   pip show opencv-python numpy tensorflow scikit-learn matplotlib seaborn
   ```

   The output should list details for each library, such as version and location.

2. **Verify `poker_calc` Installation**
   Use this command to ensure `poker_calc` is installed and accessible:

   ```bash
   pip show poker_calc
   ```

   If the library is correctly installed, it will display the version (e.g., `Version: 0.0.2`) and location.

3. **Verify Imports via Python Script**
   Run the following script to ensure all imports work as expected:

   ```bash
   python -c "import cv2; import random; import os; import numpy as np; from tensorflow.keras.models import load_model; from sklearn.metrics import confusion_matrix; import matplotlib.pyplot as plt; import seaborn as sns; from poker_calc.pokergame.pokergame import TexasHoldem; print('All imports verified successfully!')"
   ```

   If no errors occur, the setup is complete. You are now ready to proceed with the project!

## How to Run the Project

This project includes two main scripts: `main.py` and `poker.py`. Each script serves a distinct purpose and utilizes the pre-trained model for card classification.

---

#### 1. **Running the Main Classification Program**

The `main.py` script runs the core classification functionality of the project. Follow these steps:

1. **Command to Run**  
   Execute the script using the following command:

   ```bash
   python main.py
   ```

2. **Input Options**  
   The program will prompt you to enter the number of cards to test. You can choose from the following options:
   - 10 cards
   - 20 cards
   - 52 cards

3. **Functionality**  
   - The script will use a pre-trained model to classify the input cards.
   - It will display the classification results for the selected number of cards.

---

#### 2. **Running the Interactive Poker Simulation**

The `poker.py` script allows you to play a simulated poker game while integrating card classification and odds calculation. Here's how to run it:

1. **Command to Run**  
   Use the following command:

   ```bash
   python poker.py
   ```

2. **Gameplay Details**  
   - The program will simulate a poker game with the following steps:
     - Analyze card images using the pre-trained model for classification.
     - Calculate the odds of winning against a random opponent pre-flop.
   - Based on the analysis, the program will suggest whether you should:
     - Bet
     - Check
     - Fold

3. **Interactive Experience**  
   - You'll interact with the program as it guides you through the simulated poker hand, enhancing your poker strategy with data-driven insights.

---

#### Notes

- Ensure that all dependencies are installed and verified as described in the **How to Download and Verify Necessary Imports** section.
- The pre-trained model file should be in the expected location for both scripts to function correctly. Ensure that the model file is in the same directory as both script files
- The images used for training and validation can be found here: [Training Images](https://drive.google.com/drive/folders/1xwd4kDVlVuo0SylWIS8eRTN92W9zlpHQ?usp=sharing)
- For optimal performance, ensure your system meets the requirements for TensorFlow and OpenCV processing.