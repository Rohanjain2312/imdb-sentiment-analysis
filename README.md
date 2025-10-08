# IMDB Sentiment Analysis with Deep Learning

A comprehensive comparison of machine learning and deep learning models for sentiment classification on the IMDB movie review dataset, achieving up to 86.97% accuracy with Convolutional Neural Networks.

## 📊 Project Overview

This project implements and compares multiple models for binary sentiment classification (positive/negative) on IMDB movie reviews:

- **Logistic Regression (Bag-of-Words)**
- **Feed-Forward Neural Network**
- **Convolutional Neural Network (CNN)**
- **Impact Analysis of Stopword Removal**

## 🎯 Key Findings

| Model | Accuracy | Key Insight |
|-------|----------|-------------|
| **Conv1D CNN** | **86.97%** | Best performer - captures sequential patterns |
| Logistic Regression | 85.74% | Surprisingly competitive baseline |
| CNN (no stopwords) | 86.21% | Stopword removal hurts performance |
| LR (no stopwords) | 85.64% | Consistent degradation |
| Feed-Forward NN | 81.76% | Worst performer - averaging destroys context |
| FFNN (no stopwords) | 81.10% | Architecture design matters |

### Critical Insights

1. **Simple models are competitive**: Logistic Regression achieved 85.74%, only 1.23% behind CNN
2. **Architecture matters more than complexity**: Feed-Forward NN (81.76%) performed worse than Logistic Regression
3. **Keep stopwords for sentiment**: Removal degraded all models, especially CNN (-0.76%)
4. **Sequential structure is crucial**: CNNs effectively capture n-gram patterns like "not good" and "very bad"

## 🏗️ Project Structure

```
imdb-sentiment-analysis/
│
├── notebooks/
│   └── Sentiment_Analysis.ipynb          # Main analysis notebook
│
├── reports/
│   └── Performance_Analysis.md           # Detailed findings report
│
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
```

## Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
scikit-learn
numpy
pandas
matplotlib
nltk
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (for stopwords):
```python
import nltk
nltk.download('stopwords')
```

### Usage

Run the Jupyter notebook:
```bash
jupyter notebook notebooks/Sentiment_Analysis.ipynb
```

Or run directly with Python (if converted to .py):
```bash
python sentiment_analysis.py
```

## 📈 Model Architectures

### 1. Logistic Regression (Baseline)
- Binary bag-of-words features
- Vocabulary size: 5,000 words
- Solver: SAGA
- Accuracy: **85.74%**

### 2. Feed-Forward Neural Network
```python
- Embedding Layer (32 dimensions)
- GlobalAveragePooling1D
- Dense(32, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')
```
- Accuracy: **81.76%**

### 3. Convolutional Neural Network (Best)
```python
- Embedding Layer (128 dimensions)
- Conv1D(128, kernel=5) + MaxPooling1D
- Conv1D(128, kernel=5) + MaxPooling1D
- Conv1D(128, kernel=5) + MaxPooling1D
- Conv1D(256, kernel=5)
- GlobalMaxPooling1D
- Dense(64, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')
```
- Accuracy: **86.97%**
- Training time: ~352 seconds (6 epochs)

## 🔬 Methodology

### Dataset
- **Source**: IMDB Movie Review Dataset (Keras)
- **Size**: 50,000 reviews (25,000 train, 25,000 test)
- **Classes**: Binary (Positive/Negative)
- **Preprocessing**: 
  - Vocabulary limited to top 5,000 words
  - Sequences padded/truncated to 500 words

### Experimental Setup
- **GPU Usage**: Models trained on available GPU
- **Metrics**: Binary accuracy
- **Validation**: 10% of training data
- **Stopword Analysis**: Compared models with/without stopword removal

## 📊 Results Visualization

The project includes visualizations for:
- Model accuracy comparison bar charts
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrices (optional)

## 🔍 Detailed Analysis

For comprehensive analysis including:
- Architectural trade-offs
- Stopword removal impact
- Recommendations for improvement
- Path to 90%+ accuracy

See: [Performance Analysis Report](reports/Performance_Analysis.md)

## 🎓 Key Takeaways

### What Worked
- CNNs effectively capture local n-gram patterns
- Simple bag-of-words surprisingly competitive
- Keeping stopwords preserves sentiment signals
- Multiple Conv1D layers learn hierarchical features

### What Didn't Work
- Feed-Forward averaging (worst performer)
- Stopword removal (universal degradation)
- Complex architecture without proper design

### Recommendations
- **Production**: Use CNN with stopwords (86.97%)
- **Fast Baseline**: Use Logistic Regression (85.74%, CPU-only)
- **Research**: Explore LSTM/GRU, Attention, or BERT for 90%+

## 🛠️ Future Improvements

To reach 90%+ accuracy:

1. **Architecture**: Implement LSTM/GRU, Bidirectional layers, Attention mechanisms
2. **Features**: Expand vocabulary to 10k-20k words, Use pre-trained embeddings (GloVe, Word2Vec)
3. **Training**: More epochs with early stopping, Hyperparameter optimization
4. **Advanced**: Ensemble methods, Data augmentation, Fine-tune BERT/RoBERTa

## 📝 Requirements

```
tensorflow>=2.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
nltk>=3.6
```

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

##  Acknowledgments

- IMDB Dataset: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)
- TensorFlow/Keras for deep learning framework
- scikit-learn for machine learning utilities

##  References

1. Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis"
2. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification"
3. Zhang, X., & LeCun, Y. (2015). "Text Understanding from Scratch"

---

**Star ⭐ this repository if you found it helpful!**