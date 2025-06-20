import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import os
import bitsandbytes as bnb
from tqdm import tqdm
import time

from networks.model.set_transformer import SetTransformer
from networks.model.mlp import MLP as L_MLP
from diffusers import DDIMScheduler, DDPMScheduler
from Data_Set import Tensor, my_collate
from utils import draw, input_sample

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'Generator-Train-run'
format_path = 'format.svg'
train_path = '10k.pt'
sample_size = 512
learning_rate = 1e-4
BATCH_SIZE = 10240

dim_in = 6
encoded_dim = 256
number_of_strokes = 512
hidden_size = 16
steps = 4000
sample_steps = 25
beta_schedule = 'scaled_linear'
max_epochs = 10000
check_val_every_n_epoch = 10000

# Profiler settings
use_profiler = False
profile_epoch = 1  # Profile the first epoch
profile_steps = 50  # Number of steps to profile

# Set precision
torch.set_float32_matmul_precision("medium")

# Create directories
if not os.path.exists(f"Results/{experiment_name}"):
    os.makedirs(f"Results/{experiment_name}")

if not os.path.exists(f"Models/{experiment_name}"):
    os.makedirs(f"Models/{experiment_name}")

if use_profiler:
    if not os.path.exists(f"Profiler/{experiment_name}"):
        os.makedirs(f"Profiler/{experiment_name}")

# Data loading
train_set = Tensor(train_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=my_collate, pin_memory=True)

# Model initialization
model = L_MLP(
    hidden_size=hidden_size,
    hidden_layers=6,
    emb_size=64,
    time_emb="sinusoidal",
    input_emb="sinusoidal"
)

set_transformer = SetTransformer(
    dim_input=dim_in,
    num_outputs=1,
    dim_output=8,
    num_inds=8,
    dim_hidden=8,
    num_heads=4,
    ln=True
)

# Move models to device
model = model.to(device)
set_transformer = set_transformer.to(device)

# Schedulers
scheduler = DDPMScheduler(beta_end=2e-2, beta_start=1e-4, num_train_timesteps=steps, beta_schedule=beta_schedule)
ddim_s = DDIMScheduler(beta_end=2e-2, beta_start=1e-4, num_train_timesteps=steps, beta_schedule=beta_schedule)
ddim_s.set_timesteps(sample_steps)
sample_steps = list(range(25))

# Optimizer and learning rate scheduler
optimizer = bnb.optim.AdamW(list(model.parameters()) + list(set_transformer.parameters()), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[100000, 1000000, 2000000], 
    gamma=0.1
)

# Print device info
print(f"Generator model initialized on {device}")
for param in model.parameters():
    assert param.device.type == "cuda" if device == "cuda" else "cpu"
for param in set_transformer.parameters():
    assert param.device.type == "cuda" if device == "cuda" else "cpu"

def training_step(batch):
    """Single training step"""
    model.train()
    set_transformer.train()
    
    # Set Transformer Encoder
    inp = batch[0].to(device)  # [Batch, 512, 6]
    noise = torch.randn(inp.shape, device=device)  # [Batch, 512, 6]
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (inp.shape[0],), device=device).long()  # [Batch]
    
    inp = inp.transpose(0, 1)
    noise = noise.transpose(0, 1)
    noisy = scheduler.add_noise(inp, noise, timesteps)  # [Batch, 512, 6]
    noise = noise.transpose(0, 1)
    inp = inp.transpose(0, 1)
    noisy = noisy.transpose(0, 1)
    
    noisy_enc = set_transformer(noisy)  # [Batch, 512, 256]
    noisy_combined = torch.cat((noisy_enc, noisy), dim=-1)  # [Batch, 512, 262]
    noise_pred = model(noisy_combined, timesteps)  # [Batch, 512, 6]
    loss = F.mse_loss(noise_pred, noise)
    
    return loss

def validation_step(epoch):
    """Validation step - generate and save sample"""
    model.eval()
    set_transformer.eval()
    
    with torch.no_grad():
        # Generate a latent vector
        stroke = input_sample(model, set_transformer.enc, ddim_s, 6, number_of_strokes, sample_steps)
        print(f"stroke shape {stroke.shape}")
        
        # Save the generated drawing
        filename = f'Results/{experiment_name}/{epoch}.svg'
        draw(format_path, sample_size, filename, stroke)

def save_checkpoint(epoch, global_step, loss):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'set_transformer_state_dict': set_transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        'experiment_name': experiment_name
    }
    
    checkpoint_path = f"Models/{experiment_name}/epoch_{epoch:02d}-step_{global_step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Training loop
print("Starting training...")
global_step = 0
start_time = time.time()

for epoch in range(max_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    # Training loop with progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    
    # Setup profiler for the specified epoch
    if use_profiler and epoch == profile_epoch - 1:
        print(f"Starting profiling for epoch {epoch + 1}...")
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=10,
                active=profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"Profiler/{experiment_name}"),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
        profiler.start()
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Profile the training step if profiling is enabled
        if use_profiler and epoch == profile_epoch - 1 and batch_idx < profile_steps + 10:
            with record_function("training_step"):
                loss = training_step(batch)
        else:
            loss = training_step(batch)
            
        if use_profiler and epoch == profile_epoch - 1 and batch_idx < profile_steps + 10:
            with record_function("Loss Backward"):
                loss.backward()
        else:
            loss.backward()
        
        if use_profiler and epoch == profile_epoch - 1 and batch_idx < profile_steps + 10:
            with record_function("Optimizer Step"):
                optimizer.step()
        else:
            optimizer.step()
        
        
        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{epoch_loss/num_batches:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Print loss every 100 steps
        if global_step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.6f}")
        
        # Step the profiler if profiling is enabled
        if use_profiler and epoch == profile_epoch - 1:
            profiler.step()
    sort_by_keyword = "self_cpu_time_total"
    # Stop profiler if it was active
    if use_profiler and epoch == profile_epoch - 1:
        profiler.stop()
        print(profiler.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
        print(f"Profiling completed for epoch {epoch + 1}. Check Profiler/{experiment_name} for results.")
    
    # Update learning rate scheduler
    lr_scheduler.step()
    
    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
    
    # Validation step
    if (epoch + 1) % check_val_every_n_epoch == 0:
        print("Running validation...")
        validation_step(epoch + 1)
    
    # Save checkpoint every 10000 epochs (as in original)
    if (epoch + 1) % 10000 == 0:
        save_checkpoint(epoch + 1, global_step, avg_epoch_loss)

# Final validation and checkpoint
print("Training completed!")
validation_step(max_epochs)
save_checkpoint(max_epochs, global_step, avg_epoch_loss)

total_time = time.time() - start_time
print(f"Total training time: {total_time/3600:.2f} hours")

# Print profiler instructions
if use_profiler:
    print(f"\nProfiler results saved to: Profiler/{experiment_name}")
    print("To view the profiler results:")
    print("1. Install tensorboard: pip install tensorboard")
    print("2. Run: tensorboard --logdir=Profiler")
    print("3. Open http://localhost:6006 in your browser")
    print("4. Navigate to the 'PyTorch Profiler' tab")
