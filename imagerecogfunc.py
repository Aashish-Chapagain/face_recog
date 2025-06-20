import json 
import torch 
from torchvision import transforms, datasets, models
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


def train_model(data_path, epochs = 15, dataloader_batch = 16, device = "cpu", learning_rate = 0.001): 


    weights = ResNet18_Weights.DEFAULT
    transform =  weights.transforms()  

    


    dataset = datasets.ImageFolder(data_path, transform = transform)
    dataloader = DataLoader(dataset, batch_size = dataloader_batch, shuffle = True)

    model = models.resnet18(weights = weights)

    for param in model.parameters():
     param.requires_grad = False  


    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)     


    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)



    for epoch in range(epochs):
       model.train()
       running_loss = 0.0
       for images, labels in dataloader: 
           images, labels = images.to(device), labels.to(device)
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()

           print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")



    torch.save(model.state_dict(),"face_recognition_weights.pth")
    with open("classes.json", "w") as f:
        json.dump(dataset.classes, f)
    return model , dataset.classes, transform


train_model("data", epochs=15, dataloader_batch=16, device="cpu", learning_rate=0.001)