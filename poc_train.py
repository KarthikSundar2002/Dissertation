import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusers import DDPMScheduler, DDIMScheduler
import wandb
from pytorch_lightning.loggers import WandbLogger
from networks.model.POC import StrokeAttentionDiffusion, DenoiserMLP

# --- Assume the model definitions from Part 1 are in this file or imported ---

# --- Dataset Definition (from your Data_Set.py) ---
class TensorDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        # According to your prompt, each item has 5 strokes and 34 points.
        # Let's verify the shape of the loaded data.
        print(f"Loaded dataset from {path}. Number of samples: {len(self.data)}")
        if len(self.data) > 0:
            print(f"Shape of first sample: {self.data[0].shape}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Main Training Script ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    DATA_PATH = 'airplane.pt'
    EXPERIMENT_NAME = 'StrokeAttentionDiffusion_run'
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 5000 # Adjust as needed
    
    # Based on your prompt and data
    NUM_STROKES = 5
    STROKE_DIM = 34
    
    # Model specific
    ATTENTION_HEADS = 2
    ATTENTION_LAYERS = 2
    ATTENTION_DIM = 32 # Output dimension of attention, can be different from STROKE_DIM
    DENOISER_HIDDEN_SIZE = 256
    TIME_EMB_DIM = 64

    # Diffusion specific
    TRAIN_TIMESTEPS = 1000
    SAMPLING_TIMESTEPS = 25

    wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
    wandb.login(key=wand_b_key)
    wandb_logger = WandbLogger(name=EXPERIMENT_NAME,project='Your Stroke Cloud',save_dir='/scratch/ks02450')
    # --- Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium") # For performance

    # --- 1. Setup Dataset ---
    dataset = TensorDataset(DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- 2. Setup Models ---
    
    # Attention module (Transformer Encoder)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=STROKE_DIM, 
        nhead=ATTENTION_HEADS, 
        dim_feedforward=DENOISER_HIDDEN_SIZE,
        batch_first=True # Important!
    )
    attention_module = nn.TransformerEncoder(encoder_layer, num_layers=ATTENTION_LAYERS)
    # The output dimension of this module will be the same as input, i.e., STROKE_DIM
    
    # Denoiser module (MLP)
    # Input to denoiser is concatenation of noisy_stroke and attention_context
    denoiser_input_dim = STROKE_DIM + STROKE_DIM 
    denoiser_model = DenoiserMLP(
        input_size=denoiser_input_dim,
        output_size=STROKE_DIM, # It predicts the noise, which has same dim as stroke
        hidden_size=DENOISER_HIDDEN_SIZE,
        time_emb_dim=TIME_EMB_DIM
    )
    
    # --- 3. Setup Diffusion Schedulers ---
    noise_scheduler_train = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS, beta_schedule='squaredcos_cap_v2')
    noise_scheduler_sample = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS, beta_schedule='squaredcos_cap_v2')

    # --- 4. Instantiate the Main Model ---
    model = StrokeAttentionDiffusion(
        attention_module=attention_module,
        denoiser_model=denoiser_model,
        noise_scheduler=noise_scheduler_train,
        noise_scheduler_sample=noise_scheduler_sample,
        learning_rate=LEARNING_RATE,
        timesteps_for_sampling=SAMPLING_TIMESTEPS
    )

    # --- 5. Setup Trainer ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"Models/{EXPERIMENT_NAME}",
        filename="{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss"
    )

    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        logger=wandb_logger, # Uses TensorBoardLogger by default
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        profiler="simple"
    )

    # --- 6. Start Training ---
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=train_loader) # Using train_loader for validation as well for simplicity