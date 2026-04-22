import onnxruntime as ort
import numpy as np
import time
import sys

# NOTE: 'torch' is NOT imported in this file.
# This demonstrates the primary advantage of ONNX: framework independence.

def main():
    # Allow the user to specify the model path via command line arguments
    if len(sys.argv) > 1:
        onnx_model_path = sys.argv[1]
    else:
        onnx_model_path = "classifier.onnx"
    
    try:
        # Load the model using ONNX Runtime Inference Session
        # providers: Try using the GPU first, fallback to CPU
        print(f"Loading model with ONNX Runtime: '{onnx_model_path}'")
        session = ort.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"Error: Could not load the model. Please run 'python train_export.py' first.\nDetails: {e}")
        sys.exit(1)

    # Dynamically extract input information from the model
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Define the expected input shape
    # We use a batch size of 4 and feature size of 128 for inference
    batch_size = 4
    feature_size = 128
    
    print(f"Model Input Expected - Name: '{input_name}', Shape: {input_shape}")
    print(f"Running inference for {batch_size} samples...\n")
    
    # Create a completely independent input using NumPy (not a PyTorch tensor)
    dummy_input = np.random.randn(batch_size, feature_size).astype(np.float32)
    
    # Start the inference process and measure time
    start_time = time.time()
    
    # Run the session
    outputs = session.run(None, {input_name: dummy_input})
    
    end_time = time.time()
    
    # Display the results
    inference_time_ms = (end_time - start_time) * 1000
    
    print("Inference completed successfully!")
    print(f"Execution Time: {inference_time_ms:.2f} ms")
    print(f"Output Shape: {outputs[0].shape} (Batch Size x Number of Classes)")
    print("Raw predictions (Logits) for the first sample:")
    print(outputs[0][0])
    
    print("\nAs demonstrated, there is no PyTorch used in this execution!")
    print("Inference was performed solely using 'numpy' and 'onnxruntime'.")

if __name__ == "__main__":
    main()
