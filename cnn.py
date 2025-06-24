import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import torchaudio
import pandas
from torch.utils.data import DataLoader, Dataset
fixed_length = 3*16000
class ASVDataset(Dataset):
  def __init__(self, root_dir, subset='train', transform=None):

    self.root_dir=root_dir
    self.subset=subset
    self.transform=transform

    self.protocol = pandas.read_csv(
        os.path.join(root_dir, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm."+subset+".tr"+("n" if subset == 'train' else "l")+".txt"),
        sep=" ",
        header=None,
        names=["speaker", "filename", "unk1", "unk2", "label"]
    )

    self.protocol["target"] = self.protocol["label"].apply(lambda x: 1 if x == "bonafide" else 0)
  def __len__(self):
    return len(self.protocol)

  def __getitem__(self, index):
    audio_file = self.protocol.iloc[index]["filename"] + ".flac"
    audio_path = os.path.join(self.root_dir, "ASVspoof2019_LA_"+self.subset, "flac", audio_file)

    waveform, _ = torchaudio.load(audio_path)
    if waveform.size(1) > fixed_length:
      waveform = waveform[0, :fixed_length]
    else:
      waveform = F.pad(waveform, (0, fixed_length - waveform.size(1)), mode='constant', value=0)
      
    if self.transform:
      spectr = self.transform(waveform)
    else:
      spectr = waveform.unsqueeze(0)

    if spectr.dim() == 2:
        spectr = spectr.unsqueeze(0)

    label = self.protocol.iloc[index]["target"]

    return spectr, torch.tensor(label, dtype=torch.long)


class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    # Input shape: (batch, 1, n_mels=128, time_steps)
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool2 = nn. MaxPool2d(2, 2)
    
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.global_pool_time = nn.AdaptiveAvgPool2d((None, 1))

    self.fc1 = nn.Linear(256*8, 128)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(128, 2)

  # Для сохранений активаций каждого слоя
  #   self.activations = {}
  #   self._register_hooks()

  # def _register_hooks(self):
  #   def get_activation(name):
  #     def hook(model, input, output):
  #       self.activations[name] = output.detach()
  #     return hook 

    # self.conv1.register_forward_hook(get_activation('conv1'))
    # self.conv2.register_forward_hook(get_activation('conv2'))
    # self.conv3.register_forward_hook(get_activation('conv3'))
    # self.conv4.register_forward_hook(get_activation('conv4'))

  def forward(self, x):
    x = 10 * torch.log10(x + 1e-6)

    x = F.relu(self.bn1(self.conv1(x)))

    x = self.pool1(x)

    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)

    x = F.relu(self.bn3(self.conv3(x)))
    x = self.pool3(x)

    x = F.relu(self.bn4(self.conv4(x)))
    x = self.pool4(x)

    x = self.global_pool_time(x)

    x = x.squeeze(-1)

    x = x.view(x.size(0), -1)

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
  
def test(model, dataloader, criteri, device):
  total = 0
  total_loss = 0
  correct = 0
  model.eval()

  with torch.no_grad():
      for inputs, labels in dataloader:
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = criteri(outputs, labels)

          total_loss += loss.item()
          total += labels.size(0)
          _, pred_lab = torch.max(outputs.data, 1)
          correct += (pred_lab == labels).sum().item()

  return total_loss/len(dataloader), correct/total

def train(model, dataloader, criteri, optim, device):
  correct = 0
  total = 0
  total_loss =0
  model.train()

  for inputs, labels in dataloader:
      inputs, labels = inputs.to(device), labels.to(device)

      optim.zero_grad()
      outputs = model(inputs)

      loss = criteri(outputs, labels)
      loss.backward()
      optim.step()

      total_loss += loss.item()
      total += labels.size(0)
      _, pred_lab = torch.max(outputs.data, 1)
      correct += (pred_lab == labels).sum().item()

  return total_loss/len(dataloader), correct/total

def main():

  mel_transform = torchaudio.transforms.MelSpectrogram(
      sample_rate=16000,
      n_fft=1024,
      win_length=512,
      hop_length=256,
      n_mels=128,
      power=2)

# --------------------------------------------------
#                     Обучение
# --------------------------------------------------
  train_dataset = ASVDataset("/content/dataset/LA", 'train', mel_transform)
  train_dataloader = DataLoader(train_dataset, 32, shuffle=True)

  dev_dataset = ASVDataset("/content/dataset/LA", 'dev', mel_transform)
  dev_dataloader = DataLoader(dev_dataset, 32, shuffle=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN().to(device)
  print(f"Модель будет обучаться на {device}")
  criteri = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

  start = time.time()

  for epoch in range(6):
    train_loss, train_acc = train(model, train_dataloader, criteri, optim, device)
    dev_loss, dev_acc = test(model, dev_dataloader, criteri, device)
    print(f"Epoch {epoch+1}")
    print(f"Train loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
    print(f"Dev loss: {dev_loss:.4f} | Accuracy: {dev_acc*100:.2f}%")
    print("-"*50)

  end = time.time()
  torch.save(model, 'model.pth')
  print(f"Time: {(end - start):.2f} sec")

# --------------------------------------------------
#                     Тестирование
# --------------------------------------------------
  # test_dataset = ASVDataset("/content/dataset/LA", 'eval', mel_transform)
  # test_dataloader = DataLoader(test_dataset, 32, shuffle=False)

  # model = torch.load("/content/model.pth", weights_only=False)
  # criteri = nn.CrossEntropyLoss()
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # model = model.to(device)
  # print(f"Модель будет тестироваться на {device}")

  # start = time.time()

  # test_loss, test_acc = test(model, test_dataloader,criteri, device)
  # print(f"Test loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

  # end = time.time()
  # print(f"Time: {(end - start):.2f} sec")

if __name__ == "__main__":
   main()
