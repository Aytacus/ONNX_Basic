import torch
import torch.nn as nn
import os

# A simple classifier model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple neural network for demonstration purposes
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("[Step 1] Creating the model with PyTorch...")
    model = SimpleClassifier()
    
    # In a real-world scenario, the training loop would be here.
    # We are using randomly initialized weights for demonstration.
    model.eval()  # Set layers like Dropout and BatchNorm to inference mode

    # A dummy input is required for ONNX tracing
    # Batch size = 1, Features = 128
    dummy_input = torch.randn(1, 128)
    
    onnx_file_path = "classifier.onnx"
    
    print("[Step 2] Exporting the model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,        # Store the trained weights
        opset_version=17,          # Use a stable and recent opset version
        input_names=["input_tensor"], # Input name
        output_names=["output_tensor"],# Output name
        dynamic_axes={             # Enable dynamic batch sizes for future inference
            "input_tensor":  {0: "batch_size"},
            "output_tensor": {0: "batch_size"}
        }
    )
    
    print(f"Success! Model saved as '{onnx_file_path}'.")
    print("You can now run this model on any platform without needing PyTorch.")

if __name__ == "__main__":
    main()
