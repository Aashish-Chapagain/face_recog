import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights 
from torch import nn
import json
import os 


def predict_image(img_path, model, class_name, transform , device = "cpu"):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
    return class_name[predicted_idx.item()]

 

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    with open("classes.json", "r") as f:
        class_name = json.load(f)
    

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_name))
    model.load_state_dict(torch.load("face_recognition_weights.pth", map_location=device))
    model = model.to(device)


    while True:

     img_path = input("Enter the path to the image: ")
     if not os.path.exists(img_path) or not os.path.isfile(img_path) or not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.avif')):
      print("Invalid image file. Make sure the path exists, is a file, and is a supported image format.")
      continue

     img =  Image.open(img_path).convert("RGB")
     img.show()
     prediction = predict_image(img_path, model , class_name, transform, device)
     print(f"Predicted class: {prediction}")