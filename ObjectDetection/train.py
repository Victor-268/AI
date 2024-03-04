import torch
print(torch.cuda.is_available())
# !pip install detecto

from detecto import core, utils, visualize

from torchvision import transforms



custom_transforms = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize(900),
transforms.RandomHorizontalFlip(0.5),
transforms.ColorJitter(saturation=0.2),
transforms.ToTensor(),
utils.normalize_transform(),
])

Train_dataset=core.Dataset('Train/',transform=custom_transforms)#L1
Test_dataset = core.Dataset('Test/')#L2
loader=core.DataLoader(Train_dataset, batch_size=1, shuffle=True)#L3
model = core.Model(['Shoulder-good','Spine-good','Spine-bad','Shoulder-bad','Neck-good', 'Neck-bad'])#L4            #set names of classes
losses = model.fit(loader, Test_dataset, epochs=25, lr_step_size=5, learning_rate=0.002, verbose=True)#L5

model.save('model_weights.pth')

plt.plot(losses)
plt.show()


