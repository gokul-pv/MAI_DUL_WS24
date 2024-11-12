import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output mean and log variance for the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.network(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=32, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output mean and log variance for the output distribution
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = self.network(z)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu_z, logvar_z = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu_z, logvar_z)
        
        # Decode
        mu_x, logvar_x = self.decoder(z)
        
        return mu_x, logvar_x, mu_z, logvar_z
    
    def sample(self, n_samples, device, with_noise=True):
        with torch.no_grad():
            # Sample from prior p(z)
            z = torch.randn(n_samples, self.latent_dim).to(device)
            
            # Decode
            mu_x, logvar_x = self.decoder(z)
            
            if with_noise:
                # Sample from p(x|z)
                std_x = torch.exp(0.5 * logvar_x)
                eps = torch.randn_like(std_x)
                x = mu_x + eps * std_x
                return x.cpu().numpy()
            else:
                # Return mean only
                return mu_x.cpu().numpy()
