# ONNX - Escaping Framework Dependencies

This repository is designed to demonstrate the practical application of the theoretical concepts discussed in the article regarding ONNX and framework independence.

The primary goal here is to show that a model trained and built using PyTorch (or any other framework) can be executed solely via `ONNX Runtime` once exported to the `ONNX` format, without requiring PyTorch at all.

## Project Structure

The project is kept simple and divided into two main processes:
1. **`train_export.py`**: Defines a simple Neural Network (`SimpleClassifier`) using PyTorch and exports this model as `classifier.onnx`.
2. **`train_export_tf.py`**: The exact same model, but defined using TensorFlow/Keras, exported as `classifier_tf.onnx`.
3. **`inference.py`**: Completely independent of PyTorch/TensorFlow. It loads the ONNX model and performs high-speed inference using only `numpy` and `onnxruntime`.

## How to Run

### 1. Install Dependencies
Run the following command in your terminal to install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Export the Model (PyTorch or TensorFlow)
**For PyTorch:**
```bash
python train_export.py
```

**For TensorFlow:**
```bash
python train_export_tf.py
```
This command converts the model into a computation graph and generates the `.onnx` file.

### 3. Run Inference (No PyTorch or TensorFlow)
This is where the magic happens:
```bash
# Run with PyTorch's exported ONNX
python inference.py classifier.onnx

# Or run with TensorFlow's exported ONNX
python inference.py classifier_tf.onnx
```
This script takes your exported model, runs it highly optimized on the `ONNX Runtime` engine, and prints the execution time in milliseconds.

## Article and Context

To explore the story behind this project and dive deeper into the under-the-hood architecture of ONNX (Nodes, Edges, Opset), you can check out the related blog post:
[[Link to the Article]](https://medium.com/@akgunyucel/yapay-zekan%C4%B1n-evrensel-dili-onnx-ile-modelleri-%C3%B6zg%C3%BCrle%C5%9Ftirmek-c7b2299422cd)

---
**Happy Coding!**
