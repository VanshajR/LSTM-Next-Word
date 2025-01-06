
# LSTM-Next-Word


This repository contains a Streamlit app that predicts the next word in a given sentence fragment using a Long Short-Term Memory (LSTM) neural network trained on Shakespeare's *Hamlet* from the Gutenberg corpus.


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vanshajr-lstm-next-word-app-yrrb4v.streamlit.app/)

## Features

- **Interactive Web App:** A user-friendly Streamlit app to input a sentence and predict the next word.
- **Trained LSTM Model:** A deep learning model trained on the *Hamlet* text to generate contextually relevant predictions.
- **Data Processing Notebook:** A Jupyter notebook for data preprocessing, tokenization, model training, and evaluation.

## Demo

Explore the live app here: [LSTM Next Word Predictor](https://vanshajr-lstm-next-word-app-yrrb4v.streamlit.app/)

## File Structure

```
LSTM-Next-Word/
│
├── app.py               # The Streamlit app script
├── processing.ipynb     # Jupyter notebook for data processing and model training
├── requirements.txt     # List of dependencies
├── next_word_lstm.h5    # Trained LSTM model
├── tokenizer.pkl        # Tokenizer object for word-index mapping
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/VanshajR/LSTM-Next-Word.git
   cd LSTM-Next-Word
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app locally:**

   ```bash
   streamlit run app.py
   ```

## Dataset

The app uses the text of *Hamlet* by William Shakespeare, sourced from the Gutenberg corpus (`nltk.corpus.gutenberg`).

## How It Works

1. **Data Preprocessing:**
   - The *Hamlet* text is tokenized, and sequences of words are generated.
   - Input sequences are padded to uniform lengths, and labels are one-hot encoded.

2. **Model Architecture:**
   - An embedding layer maps words to dense vectors.
   - Two LSTM layers capture temporal dependencies in the text.
   - A dense softmax layer outputs the probabilities of the next word.

3. **Streamlit App:**
   - Users input a sentence fragment.
   - The app tokenizes the input, pads it, and uses the LSTM model to predict the next word.

## Example

```python
Input: "To be or not to be"
Output: "buried"
```

## Training

Model training is detailed in `processing.ipynb`. It includes:
- Splitting data into training and test sets.
- Training the LSTM model for 80 epochs.
- Saving the trained model and tokenizer.

## Dependencies

The project uses the following Python libraries:
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `streamlit`

All dependencies are listed in `requirements.txt`.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

