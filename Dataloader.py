def downloadSampling(img):
    image = np.array(img)
    image_blur = cv2.resize(image,(64,64),cv2.INTER_CUBIC)
    new_image = Image.fromarray(image_blur)
    return new_image

HR_transform = transforms.Compose([
                                 transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
LR_transform = transforms.Compose([
                                   transforms.Resize((256,256)),
                                   transforms.Lambda(downloadSampling),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

#from torch.utils.data.dataloader import T
import torchvision
from torchvision import datasets
#LR_train_dataset = datasets.STL10(root = "/content/drive/MyDrive/DIV2K_train_HR",transform = HR_transform, download = False)
LR_train_dataset = datasets.ImageFolder(root = "/content/drive/MyDrive/Dataset.div2k",transform = LR_transform)
LR_train_dataloader = DataLoader(LR_train_dataset, batch_size = 5)
#HR_train_dataset = datasets.STL10(root = "/content/drive/MyDrive/DIV2K_train_HR",transform = HR_transform, download = False)
HR_train_dataset = datasets.ImageFolder(root = "/content/drive/MyDrive/Dataset.div2k",transform = HR_transform)
HR_train_dataloader = DataLoader(HR_train_dataset, batch_size = 5)

print(len(LR_train_dataset))

HR_batch = next(iter(HR_train_dataloader))
plt.figure(figsize=(20,20))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(HR_batch[0].to(device)[:8], padding=2, normalize=True).cpu(),(1,2,0)))