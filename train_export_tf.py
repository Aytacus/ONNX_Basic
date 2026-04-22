import tensorflow as tf
import tf2onnx
import os

def main():
    print("[Step 1] Creating the model with TensorFlow/Keras...")
    # Create the equivalent sequential model to our PyTorch one
    # Input size: 128 -> Hidden layer 1: 64 (ReLU) -> Output layer: 10
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10) # Logits output without softmax to match PyTorch implementation
    ])

    onnx_file_path = "classifier_tf.onnx"
    
    print("[Step 2] Exporting the model to ONNX format using tf2onnx...")
    
    # Define the input signature specifying dynamic batch size (None) and static feature size (128)
    # Naming the input 'input_tensor' to match the PyTorch export exactly
    input_signature = [tf.TensorSpec([None, 128], tf.float32, name="input_tensor")]
    
    # Keras 3 (TensorFlow 2.16+) has a known issue with tf2onnx.convert.from_keras.
    # To bypass this, we wrap the model call in a pure TensorFlow function.
    @tf.function(input_signature=input_signature)
    def inference_fn(x):
        return model(x)
    
    # Convert from TensorFlow function to ONNX
    onnx_model, _ = tf2onnx.convert.from_function(
        inference_fn, 
        input_signature=input_signature, 
        opset=17
    )

    # Save the ONNX model to disk
    with open(onnx_file_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Success! Model saved as '{onnx_file_path}'.")
    print("You can now run this TensorFlow model in 'inference.py' exactly the same way!")

if __name__ == "__main__":
    main()
