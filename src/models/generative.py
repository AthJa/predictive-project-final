import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TadGANDataset(Dataset):
    def __init__(self, X, sequence_length):
        self.X = X
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx:idx+self.sequence_length], dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z):
        x = self.fc(z)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out

class CriticX(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticX, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class CriticZ(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(CriticZ, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)

def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    if len(real.shape) == 3:
        alpha = torch.rand(batch_size, 1, 1).to(device)
    else:
        alpha = torch.rand(batch_size, 1).to(device)
        
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolates = critic(interpolates)
    
    fake_targets = torch.ones(batch_size, 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_targets,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.reshape(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_tadgan(train_loader, input_dim, seq_len, latent_dim=20, hidden_dim=64, epochs=10, lr=0.0005, n_critic=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
    generator = Generator(latent_dim, hidden_dim, input_dim, seq_len).to(device)
    cx = CriticX(input_dim, hidden_dim).to(device)
    cz = CriticZ(latent_dim, hidden_dim).to(device)
    
    opt_e = torch.optim.Adam(encoder.parameters(), lr=lr)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_cx = torch.optim.Adam(cx.parameters(), lr=lr)
    opt_cz = torch.optim.Adam(cz.parameters(), lr=lr)
    
    mse = nn.MSELoss()
    
    for epoch in range(epochs):
        cx_loss_epoch = 0
        cz_loss_epoch = 0
        eg_loss_epoch = 0
        
        for i, x in enumerate(train_loader):
            batch_size = x.size(0)
            x = x.to(device)
            
            # --- 1. Train Critics (Cx and Cz) ---
            for _ in range(n_critic):
                z_real = torch.randn(batch_size, latent_dim).to(device)
                z_fake = encoder(x).detach()
                x_fake = generator(z_real).detach()
                
                # Critic X
                cx_real = cx(x)
                cx_fake = cx(x_fake)
                gp_x = gradient_penalty(cx, x, x_fake, device)
                loss_cx = -(torch.mean(cx_real) - torch.mean(cx_fake)) + 10.0 * gp_x
                
                opt_cx.zero_grad()
                loss_cx.backward()
                opt_cx.step()
                
                # Critic Z
                cz_real = cz(z_real)
                cz_fake = cz(z_fake)
                gp_z = gradient_penalty(cz, z_real, z_fake, device)
                loss_cz = -(torch.mean(cz_real) - torch.mean(cz_fake)) + 10.0 * gp_z
                
                opt_cz.zero_grad()
                loss_cz.backward()
                opt_cz.step()
                
                cx_loss_epoch += loss_cx.item()
                cz_loss_epoch += loss_cz.item()
                
            # --- 2. Train Encoder & Generator ---
            z_fake = encoder(x)
            x_fake = generator(torch.randn(batch_size, latent_dim).to(device))
            x_rec = generator(z_fake)
            
            cx_fake_g = cx(x_fake)
            cz_fake_e = cz(z_fake)
            
            loss_e = -torch.mean(cz_fake_e)
            loss_g = -torch.mean(cx_fake_g)
            loss_rec = mse(x, x_rec) * 10.0
            
            loss_eg = loss_e + loss_g + loss_rec
            
            opt_e.zero_grad()
            opt_g.zero_grad()
            loss_eg.backward()
            opt_e.step()
            opt_g.step()
            
            eg_loss_epoch += loss_eg.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Cx: {cx_loss_epoch/(len(train_loader)*n_critic):.4f} | Cz: {cz_loss_epoch/(len(train_loader)*n_critic):.4f} | EG: {eg_loss_epoch/len(train_loader):.4f}")
        
    return encoder, generator, cx, cz

def calculate_anomaly_scores(encoder, generator, cx, dataloader, alpha=0.5):
    """
    Combines Reconstruction Error and Critic Score.
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    generator.eval()
    cx.eval()
    
    scores = []
    reconstructions = []
    
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            z = encoder(x)
            x_rec = generator(z)
            
            rec_err = torch.mean((x - x_rec)**2, dim=[1, 2])
            critic_score = cx(x).squeeze(-1)
            
            scores.append(torch.stack([rec_err, critic_score], dim=1).cpu().numpy())
            reconstructions.append(x_rec.cpu().numpy())
            
    scores = np.concatenate(scores, axis=0)
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    rec_norm = (scores[:, 0] - np.min(scores[:, 0])) / (np.max(scores[:, 0]) - np.min(scores[:, 0]) + 1e-8)
    crit_norm = (np.max(scores[:, 1]) - scores[:, 1]) / (np.max(scores[:, 1]) - np.min(scores[:, 1]) + 1e-8)
    
    final_scores = alpha * rec_norm + (1 - alpha) * crit_norm
    return final_scores, reconstructions
