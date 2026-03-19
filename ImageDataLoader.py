from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class ImageDataLoader:
   
    def __init__(self,size,path):
        self.batch_size = size
        self.data_path = path

     
    def LoadDataset(self):
       
        preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ]) #normalize/preprocess for images

        dataset = datasets.ImageFolder(root = self.data_path, transform=preprocess) #automatically assigns label for waldo and notwaldo folders (waldo = 1, notwaldo = 0)

        return dataset

    def LoadBatch(self,dataset):
        load_data = DataLoader(dataset,batch_size = self.batch_size, shuffle=True) 

        return load_data