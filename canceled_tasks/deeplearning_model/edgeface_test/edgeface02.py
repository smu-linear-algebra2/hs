import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

# Load model
model01 = "edgeface_xs_gamma_06"  # or edgeface_s_gamma_05
model = get_model(model01)
checkpoint_path = f'checkpoints/{model01}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Path to the face image
path = 'C:/smu/linear_algebra/data/yongan'
aligned = align.get_aligned_face(path)  # Align face
transformed_input = transform(aligned)  # Preprocess

# Extract embedding
embedding = model(transformed_input.unsqueeze(0))