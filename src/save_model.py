import torch
from model import DiabetesNet

# Define model parameters
input_dim = 8
model = DiabetesNet(input_dim)

# ❗️FIXED: Provide the full path to the file, not just the directory.
# Added map_location to ensure it works on CPU-only machines.
model_path = "checkpoints/diabetes_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode before saving
model.eval()

# Save the entire model (architecture + weights)
torch.save(model, "diabetes_model_full.pt")

print("✅ Model saved successfully to 'diabetes_model_full.pt'!")