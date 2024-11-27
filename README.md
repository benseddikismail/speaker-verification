# Speaker Verification System

This project implements a **Speaker Verification System** using a **Siamese Neural Network** to determine whether two audio utterances are from the same speaker or different speakers.

## Overview

The system performs speaker verification, where the goal is to classify whether two audio samples (utterances) come from the same speaker (positive class) or from different speakers (negative class). It uses a **Siamese Network** to process pairs of audio samples and predict their similarity.

### Dataset

The dataset consists of two parts:
- **Training Data** (`trs.pkl`): A 500 × 16,180 matrix containing 50 speakers, each with 10 utterances.
- **Test Data** (`tes.pkl`): A 200 × 22,631 matrix containing 20 speakers, each with 10 utterances.

## Method

### 1. Pair Generation

- **Positive Pairs**: Pairs of utterances from the same speaker.
- **Negative Pairs**: Pairs of utterances from different speakers.
  
For each minibatch, there are `L` pairs of each type, and the total number of pairs per batch is `2L` (with equal numbers of positive and negative examples).

### 2. Siamese Network

The network architecture processes two spectrograms of size \( 513 \times T \), where `T` is the number of time steps in the STFT representation. The network outputs fixed-length feature vectors for each input, and their inner product is passed through a sigmoid activation to predict the class (same or different speaker).

**Loss Function**: Binary Cross-Entropy.

### 3. Training Process

- The training data is divided into positive and negative pairs, forming minibatches.
- The model is trained on these pairs to classify whether the utterances belong to the same speaker or not.

### 4. Testing and Evaluation

- The test set is used to form pairs of utterances, and the model is evaluated on its accuracy for classifying speaker pairs.
