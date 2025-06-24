import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import torchaudio
import pandas
from torch.utils.data import DataLoader, Dataset
import math
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

    spectr = torch.log(spectr + 1e-8)

    if spectr.dim() == 3:
        spectr = spectr.squeeze(0)

    label = self.protocol.iloc[index]["target"]

    return spectr, torch.tensor(label, dtype=torch.long)

class PositionEncodeing(nn.Module):
  def __init__(self, d_model, max_len=500):
    super().__init__()

    pe = torch.zeros(max_len, d_model)

    # Массив позиций в seq_len
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # Находим различные значения в зависимости от разных измерений d_model
    div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))

    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)

    self.register_buffer('pe', pe)
  
  def forward(self, x):
    pe = self.pe[:x.size(0), :]
    pe = pe.unsqueeze(1)
    x = x + pe
    return x
  
class Transformer(nn.Module):
  def __init__(self, n_mels=64, d_model=128, n_head=8, num_layers=4):
    super().__init__()
    self.input = nn.Linear(n_mels, d_model)

    self.pos_encoder = PositionEncodeing(d_model)

    encoder_layers = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=n_head,
      dim_feedforward=512,
      dropout=0.1
    )
    self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

    self.classifier = nn.Sequential(
      nn.Linear(d_model, 64),
      nn.ReLU(),
      nn.Linear(64, 2)
    )

  def forward(self, x):
    # x: (batch, n_mels, time) -> (time, batch, n_mels)
    x = x.permute(2, 0, 1)

    x = self.input(x)
    x = self.pos_encoder(x)

    x = self.transformer(x) # (time, batch, d_model)

    x = x.mean(dim=0)

    return self.classifier(x)
    

  
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
      n_mels=64,
      )

# --------------------------------------------------
#                     Education
# --------------------------------------------------
  train_dataset = ASVDataset("/content/dataset/LA", 'train', mel_transform)
  train_dataloader = DataLoader(train_dataset, 32, shuffle=True)

  dev_dataset = ASVDataset("/content/dataset/LA", 'dev', mel_transform)
  dev_dataloader = DataLoader(dev_dataset, 32, shuffle=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Transformer().to(device)
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
  torch.save(model, 'model_transformer.pth')
  print(f"Time: {(end - start):.2f} sec")

# --------------------------------------------------
#                     Testing
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
