import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx:idx+self.sequence_length], dtype=torch.float32),
                torch.tensor(self.y[idx+self.sequence_length], dtype=torch.float32))

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def tune_sequence_length(X, y, lengths=[8, 16, 48, 96], epochs=5, batch_size=32, dropout=0.0):
    """
    Trains models on different sequence lengths and selects the one that minimizes MAE on validation.
    """
    best_length = lengths[0]
    best_mae = float('inf')
    
    # Split for tuning (simple 80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seq_len in lengths:
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = BiLSTMModel(input_dim=X.shape[1], hidden_dim=64, num_layers=2, dropout=dropout).to(device)
        criterion = nn.L1Loss() # MAE
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                
        # Evaluate on validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_mae = val_loss / len(val_dataset)
        print(f"Sequence Length: {seq_len}, Val MAE: {val_mae:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_length = seq_len
            
    print(f"Best Sequence Length: {best_length} with MAE: {best_mae:.4f}")
    return best_length

def train_bilstm(train_loader, val_loader, input_dim, hidden_dim=64, num_layers=2, epochs=10, lr=0.001, dropout=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    
    criterion = nn.L1Loss() # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Val MAE: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    print(f"Training complete. Best Val MAE: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model

def predict_bilstm(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            
            # handle batch size 1 correctly
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
            
    return np.array(predictions), np.array(actuals)

def evaluate_bilstm(model, dataloader):
    from sklearn.metrics import mean_absolute_error, r2_score
    
    predictions, actuals = predict_bilstm(model, dataloader)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {'MAE': mae, 'R2': r2}

if __name__ == "__main__":
    pass
