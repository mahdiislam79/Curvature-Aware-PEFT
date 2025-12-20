# Import necessary libraries
import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

import torch
import gc  # Garbage collector for memory management
import torch.nn as nn
import math

from typing import Dict, Optional, Tuple, List
from torch.optim import Optimizer

import nltk
nltk.download('punkt', quiet=True) # For evaluation metrics like ROUGE


from peft.tuners.lora import LoraLayer # The base LoRA layer class from PEFT
from transformers import (
    Seq2SeqTrainer,    # The base Hugging Face Trainer
    TrainerCallback, # The base Callback class
)

# === 1. The K-FAC Data Collector ===
# This is the most low-level part. It's a custom PyTorch function
# that intercepts the model's forward and backward passes.
class KFACAutogradFunction(torch.autograd.Function):
    """Custom autograd function to capture activations and gradients for K-FAC."""

    @staticmethod
    def forward(ctx, module, output, input):
        # 'ctx' is a context object to save data for the backward pass.
        # This function runs during the model's FORWARD pass.
        ctx.module = module  
        # .detach() means we don't need to track gradients for this saved tensor.
        ctx.save_for_backward(input.detach())
        return output 

    @staticmethod
    def backward(ctx, grad_wrt_output):
        # 'grad_wrt_output' is the gradient 'g' flowing back into this layer.
        # Retrieve the saved data
        module = ctx.module
        input, = ctx.saved_tensors # This is the input 'a' we saved in forward()
        manager = module.grit_manager 

        # --- Check if we should update K-FAC on this step ---
        if not module.training:
            return None, grad_wrt_output, None 

        # This is a 'tick' counter for each module to control update frequency.
        tick = manager.cov_update_tick.get(module, 0) + 1
        manager.cov_update_tick[module] = tick
        
        # We DON'T update K-FAC factors every step. It's too slow. We only do it based on 'grit_cov_update_freq'.
        if tick % manager.config.grit_cov_update_freq != 0:
            return None, grad_wrt_output, None 

        # --- Perform K-FAC Covariance Update ---
        with torch.no_grad(): 
            
            # --- 1. Activation Covariance (A-Cov) ---
            # 'a' is the input activation, reshaped into a 2D matrix
            a = input.reshape(-1, input.shape[-1])
            if a.shape[0] > 0 and 'default' in module.lora_A:
                
                grad_device = grad_wrt_output.device  

                # Project 'a' using the LoRA A matrix's transpose. This gives us the activations *in the LoRA rank space*.
                lora_A_T = module.lora_A['default'].weight.data.T.to(device=grad_device, dtype=a.dtype, non_blocking=True)
                projected_a = a @ lora_A_T
                
                # Calculate the sample covariance: (a^T * a). This is the "Kronecker factor" for activations.
                a_cov_sample = projected_a.T @ projected_a

                # Get the long-term running average of the covariance
                current_cov = manager.a_covs[module].to(grad_device)
                n = manager.num_samples_a[module]
                new_n = n + projected_a.shape[0]
                if new_n > 0:
                    # Update the running average: (old_avg * old_n + new_sample) / new_n
                    updated_cov = (current_cov.float() * n + a_cov_sample.float()) / new_n
                    manager.a_covs[module].copy_(updated_cov.to(dtype=manager.a_covs[module].dtype))
                    manager.num_samples_a[module] = new_n

            # --- 2. Gradient Covariance (G-Cov) ---
            # 'g' is the output gradient, reshaped into a 2D matrix
            g = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
            if g.shape[0] > 0 and 'default' in module.lora_B:
                # Project 'g' using the LoRA B matrix. This gives us the gradients *in the LoRA rank space*.
                lora_B = module.lora_B['default'].weight.data.to(device=g.device, dtype=g.dtype, non_blocking=True)
                projected_g = g @ lora_B
                
                # Calculate the sample covariance: (g^T * g). This is the "Kronecker factor" for gradients.
                g_cov_sample = projected_g.T @ projected_g

                # Update the running average for the 'g' covariance, just like 'a'
                current_cov = manager.g_covs[module].to(g_cov_sample.device)
                n = manager.num_samples_g[module]
                new_n = n + projected_g.shape[0]
                if new_n > 0:
                    updated_cov = (current_cov.float() * n + g_cov_sample.float()) / new_n
                    manager.g_covs[module].copy_(updated_cov.to(dtype=manager.g_covs[module].dtype))
                    manager.num_samples_g[module] = new_n

        # We return the original gradients unchanged. The 'None's correspond to the other inputs of the forward() function (module, output, input).
        return None, grad_wrt_output, None


# === 2. The GRIT Configuration ===
class GRITConfig:
    """Configuration for GRIT training on Llama models."""

    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.batch_size = 1 
        self.gradient_accumulation_steps = 8
        self.num_epochs = 3
        self.learning_rate = 1e-5
        self.precision = "bf16"
        self.max_length = 1024 
        # self.min_pixels = 256 * 28 * 28 # For Qwen3's dynamic image resolution
        # self.max_pixels = 1280 * 28 * 28

        # --- LoRA Configs ---
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        # Target modules for Qwen3-VL (language and vision parts)
        self.lora_target_modules = [
            # Language & Vision Attention (q_proj, k_proj, v_proj, o_proj)
            "q_proj", "k_proj", "v_proj", "o_proj",
            # Language MLP (gate_proj, up_proj, down_proj)
            "gate_proj", "up_proj", "down_proj",
            # Vision MLP (fc1, fc2)
            "fc1", "fc2"
        ]
        self.apply_lora_to_vision = False
        self.vision_lora_modules = []

        # --- GRIT K-FAC Configs ---
        self.kfac_update_freq = 1         # How often (in manager steps) to invert K-FAC
        self.kfac_damping = 0.005         # Stability term for matrix inversion
        self.kfac_min_samples = 1         # Min samples before trying to invert
        self.grit_cov_update_freq = 2     # How often (in backward steps) to update covs

        # --- GRIT Reprojection Configs ---
        self.reprojection_freq = 2        # How often (in manager steps) to run reprojection
        self.reprojection_k = 2           # A parameter for reprojection (not used if rank_adapt is on)
        self.use_two_sided_reprojection = True # Use both A-Cov and G-Cov for reprojection

        # --- GRIT Rank Adaptation Configs ---
        self.enable_rank_adaptation = True # The main switch for adaptive rank
        self.rank_adaptation_threshold = 0.99 # Keep the smallest rank 'k' that captures 99% of the "energy"
        self.min_lora_rank = 4            # Don't adapt rank below this value

        # --- Warmup Configs ---
        self.regularizer_warmup_steps = 100 # Slowly ramp up the GRIT loss penalties
        self.reprojection_warmup_steps = 200 # Don't start reprojection until step 200
        self.rank_adaptation_start_step = 200 # Don't start *adapting* rank until step 200
        self.ng_warmup_steps = 100          # Don't apply natural gradient (preconditioning) until step 100

        # --- Dataset and Dataloader ---
        self.dataset_name = "ChongyanChen/VQAonline"
        self.num_workers = 2
        self.pin_memory = True
        self.drop_last = True 

        # --- GRIT Loss Lambdas (Penalties) ---
        self.lambda_kfac = 5e-6   # Strength of the curvature regularization
        self.lambda_reproj = 5e-5 # Strength of the reprojection regularization

        # --- Logging Configs ---
        self.log_fisher_spectrum = True # ... various logging flags ...
        self.log_top_eigs = 8
        self.log_eig_heatmaps = True
        self.log_eigs_bar = False
        self.log_eig_heatmaps_modules = 4
        self.log_eff_rank_on_inversion = False
        self.log_final_eff_rank = True
        self.kfac_inversion_device = 'cpu' # Do expensive inversions on CPU to save VRAM

        # --- VQA Configs ---
        self.answer_max_length = 128
        self.ignore_index = -100 # Standard for labels

        # --- Optimizer / memory ---
        self.use_8bit_adam = True          # Use 8-bit Adam to save memory
        self.gradient_checkpointing = True # Trade compute for VRAM

        # --- Eval / generation ---
        self.gen_max_new_tokens = 128
        self.gen_do_sample = False


# === 3. The GRIT "Brain" (Manager) ===
class GRITManager:
    """Manager for GRIT on Llama models."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.global_step = 0 # Counter for trainer steps
        self.backward_step = 0 # Counter for backward passes
        
        # --- State Tracking ---
        self.loss_history = [] # For adaptive frequency
        self.loss_history_capacity = 20
        self.last_kfac_update_step = 0
        self.last_reprojection_step = 0
        
        # --- K-FAC Storage ---
        # These dictionaries will map [LoRA_Module -> Tensor]
        self.a_covs = {}          # Activation covariances (A_cov)
        self.g_covs = {}          # Gradient covariances (G_cov)
        self.a_invs = {}          # INVERTED A_cov
        self.g_invs = {}          # INVERTED G_cov
        self.num_samples_a = {}   # Number of samples in A_cov running avg
        self.num_samples_g = {}   # Number of samples in G_cov running avg
        self.cov_update_tick = {} # Per-module step counter
        
        self.optimized_modules = [] # List of all LoRA modules we are managing
        self.factors_are_ready = False # Flag to tell optimizer if inverses are ready
        
        # --- Reprojection Storage ---
        self._last_Va_k = {} # Caches the "important subspace" for 'A'
        self._last_Vg_k = {} # Caches the "important subspace" for 'B'
        
        # --- Start the process ---
        self._instrument_model()
        print("GRITManager: Initialization complete for Llama-3.2-11B model.")
        print(f"Optimizing {len(self.optimized_modules)} LoRA modules across vision + language towers.")

    def _instrument_model(self):
        """Finds all LoRA layers and "injects" the K-FAC autograd function."""
        print("Instrumenting model for GRIT...")
        vision_modules = 0
        language_modules = 0

        # Loop over every single module in the entire model
        for name, module in self.model.named_modules():
            # Check if it's a LoRA layer we care about (is active, is Linear)
            if isinstance(module, LoraLayer) and module.r.get('default', 0) > 0 and isinstance(module.base_layer, nn.Linear):
                module.lora_name = name      # Store its name for logging
                module.grit_manager = self   # Give it a handle to the manager

                # --- Detect if it's a Vision or Language module ---
                # (This is just for logging, not for GRIT logic)
                name_lower = name.lower()
                is_vision = name.startswith('model.vision_model')

                # --- Initialize K-FAC state for this module ---
                self.num_samples_a[module] = 0
                self.num_samples_g[module] = 0
                self.cov_update_tick[module] = 0

                # --- THE GRIT adaptation ---
                # 1. Save the original forward function
                module.original_forward = module.forward
                
                # 2. Define a new forward function
                def new_forward(self, x=None, *args, **kwargs):
                    # (Handle Qwen's flexible input anmes)
                    if x is None:
                        x = kwargs.get("hidden_states", None)
                        if x is None and len(args) > 0:
                            x = args[0]
                    # 3. Call the original forward function
                    y = self.original_forward(x, *args, **kwargs)
                    
                    # 4. WRAP the call with our custom KFAC function
                    # This "hooks" the layer, enabling our custom backward pass.
                    return KFACAutogradFunction.apply(self, y, x)

                # 5. Replace the module's forward with our new one
                module.forward = new_forward.__get__(module, LoraLayer)

                # --- Initialize empty covariance matrices ---
                weight_device = module.lora_A['default'].weight.device  # ← Get from actual weights
                r = module.r['default'] # The LoRA rank
                cov_device = self.device if torch.cuda.is_available() else torch.device('cpu')
                self.a_covs[module] = torch.zeros((r, r), device=weight_device, dtype=torch.float16)
                self.g_covs[module] = torch.zeros((r, r), device=cov_device, dtype=torch.float16)
                self.optimized_modules.append(module) # Add to our list

                if is_vision:
                    vision_modules += 1
                else:
                    language_modules += 1

        print(f"Instrumented {vision_modules} vision modules + {language_modules} language modules")
        print(f"Total LoRA modules with K-FAC: {len(self.optimized_modules)}")    

    def _compute_effective_rank(self, cov: torch.Tensor) -> float:
        """Helper function to calculate entropy-based effective rank for logging."""
        try:
            with torch.no_grad():
                evals = torch.linalg.eigvalsh(cov.float()) # Get eigenvalues
                evals = torch.clamp(evals, min=0.0)
                total = float(evals.sum().item())
                if not math.isfinite(total) or total <= 0.0:
                    return 0.0
                p = (evals / total).clamp_min(1e-12) # Normalize to a probability distribution
                entropy = float((-(p * torch.log(p)).sum()).item()) # Calculate Shannon entropy
                r_eff = math.exp(entropy) # Effective rank is exp(entropy)
                n = int(cov.shape[0])
                if not math.isfinite(r_eff):
                    return 0.0
                return float(max(0.0, min(n, r_eff)))
        except Exception:
            return 0.0

    def _get_adaptive_freq(self, base_freq, min_freq=1, max_freq=1000, window=20):
        """Dynamically adjusts update frequency based on loss behavior."""
        if len(self.loss_history) < window:
            return base_freq # Not enough history, use default
        
        # Check if loss is decreasing
        recent_losses = self.loss_history[-window:]
        first_half = sum(recent_losses[:window//2]) / (window//2)
        second_half = sum(recent_losses[window//2:]) / (window//2)
        
        if second_half < first_half * 0.99: # If loss is dropping...
            new_freq = int(base_freq * 1.5) # ...slow down (do GRIT less often)
        else: # If loss is stagnating or increasing...
            new_freq = int(base_freq * 0.75) # ...speed up (do GRIT more often)
        
        return max(min_freq, min(new_freq, max_freq)) # Clamp within bounds

    def step(self, loss=None):
        """This is the main "tick" function, called by GritCallback on each trainer step."""
        self.global_step += 1
        
        # --- Update Loss History (for adaptive freq) ---
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.loss_history_capacity:
                self.loss_history = self.loss_history[-self.loss_history_capacity:]

        # --- Adapt K-FAC Damping (for stability) ---
        if len(self.loss_history) > 10:
            loss_variance = torch.tensor(self.loss_history).var().item()
            # If loss is volatile, increase damping. If stable, decrease it.
            self.config.kfac_damping = max(1e-6, min(0.01, 0.001 + math.sqrt(loss_variance)))

        # --- Decide: Time to update K-FAC inverses? ---
        kfac_freq = self._get_adaptive_freq(self.config.kfac_update_freq)
        if self.global_step - self.last_kfac_update_step >= kfac_freq:
            self.update_and_invert_factors() # This is the expensive step
            self.last_kfac_update_step = self.global_step

        # --- Decide: Time to do Neural Reprojection? ---
        reproj_freq = self._get_adaptive_freq(self.config.reprojection_freq, min_freq=1, max_freq=2000)
        if (self.global_step >= int(getattr(self.config, 'reprojection_warmup_steps', 0) or 0) and
           self.global_step - self.last_reprojection_step >= reproj_freq):
            self.neural_reprojection() # This is the rank adaptation step
            self.last_reprojection_step = self.global_step

    def update_and_invert_factors(self):
        """The expensive step: Invert all A_cov and G_cov matrices."""
        print(f"\nGRITManager: Inverting K-FAC factors at step {self.global_step}...")
        
        # Compile the inversion function for speed
        if not hasattr(self, "_scripted_invert_fn") or self._scripted_invert_fn is None:
            self._scripted_invert_fn = torch.jit.script(jit_invert_tensor_pair)
        scripted_invert_fn = self._scripted_invert_fn

        for module in self.optimized_modules:
            # Wait until we have enough data samples
            min_samples = getattr(self.config, 'kfac_min_samples', 128)
            if self.num_samples_a.get(module, 0) < min_samples or self.num_samples_g.get(module, 0) < min_samples:
                continue # Skip this module

            # --- Offload inversion to CPU to save VRAM ---
            inv_device = str(getattr(self.config, 'kfac_inversion_device', 'cpu') or 'cpu')
            if inv_device == 'cuda' and torch.cuda.is_available():
                device_for_inv = torch.device('cuda')
            else:
                device_for_inv = torch.device('cpu')

            a_cov = self.a_covs[module].detach().to(device=device_for_inv, dtype=torch.float32)
            g_cov = self.g_covs[module].detach().to(device=device_for_inv, dtype=torch.float32)

            # Call the JIT-compiled, numerically stable inverter
            a_inv, g_inv = scripted_invert_fn(a_cov=a_cov, g_cov=g_cov, kfac_damping=self.config.kfac_damping)
            
            if a_inv.numel() > 0 and g_inv.numel() > 0:
                # Store the inverses (on CPU)
                self.a_invs[module] = a_inv.to('cpu')
                self.g_invs[module] = g_inv.to('cpu')

        self.factors_are_ready = True # Set flag: OK to start preconditioning!

    def log_final_effective_ranks(self):
        """Logging function to run at the end of training."""
        if not getattr(self.config, 'log_final_eff_rank', True):
            return
        print("--- Final Effective Ranks ---")
        try:
            for module in self.optimized_modules:
                a_cov = self.a_covs[module].detach().to(device='cpu', dtype=torch.float32)
                g_cov = self.g_covs[module].detach().to(device='cpu', dtype=torch.float32)
                r_eff_a = self._compute_effective_rank(a_cov)
                r_eff_g = self._compute_effective_rank(g_cov)
                print(f"Fisher/{module.lora_name}/eff_rank_a: {r_eff_a:.4f}")
                print(f"Fisher/{module.lora_name}/eff_rank_g: {r_eff_g:.4f}")
        except Exception:
            pass # Don't crash on logging

    def log_final_raw_ranks(self):
        """Logs the final *adapted* rank 'k' for each module."""
        print("--- Final Adapted Ranks (k) ---")
        try:
            for module in self.optimized_modules:
                base_r = int(module.r['default']) if hasattr(module, 'r') and 'default' in module.r else 0
                Va = getattr(self, '_last_Va_k', {}).get(module, None)
                Vg = getattr(self, '_last_Vg_k', {}).get(module, None)
                k_a = int(Va.shape[1]) if Va is not None and Va.numel() > 0 else base_r
                k_b = int(Vg.shape[1]) if Vg is not None and Vg.numel() > 0 else k_a
                print(f"Fisher/{module.lora_name}/final_k_a: {k_a}")
                print(f"Fisher/{module.lora_name}/final_k_b: {k_b}")
        except Exception:
            pass

    def precondition_gradients(self):
        """THE PAYOFF: Modifies gradients before the optimizer step."""
        if not self.factors_are_ready: # Don't run if inverses aren't ready
            return

        with torch.no_grad():
            # Check for NG (Natural Gradient) warmup
            ng_warmup = int(getattr(self.config, 'ng_warmup_steps', 0) or 0)
            if self.global_step < ng_warmup:
                return # Still warming up

            _total_nat_norm_sq = torch.tensor(0.0, device=self.device)

            for module in self.optimized_modules:
                if module not in self.a_invs or module not in self.g_invs:
                    continue # Inverses not ready for this module

                lora_a = module.lora_A['default']
                lora_b = module.lora_B['default']

                if (lora_a is None or lora_b is None or
                    lora_a.weight.grad is None or lora_b.weight.grad is None):
                    continue # No gradients to precondition

                # NEW CODE - CORRECT
                grad_device = lora_a.weight.grad.device

                # --- Load inverses (from CPU) to the GPU ---
                a_inv_f32 = self.a_invs[module].to(grad_device, dtype=torch.float32)
                g_inv_f32 = self.g_invs[module].to(self.device, dtype=torch.float32)

                # --- THIS IS THE NATURAL GRADIENT STEP ---
                # grad_B_ng = grad_B @ G_inv
                grad_b_f32 = lora_b.weight.grad.detach().to(torch.float32)
                preconditioned_b_grad_f32 = grad_b_f32 @ g_inv_f32

                # grad_A_ng = A_inv @ grad_A
                grad_a_f32 = lora_a.weight.grad.detach().to(torch.float32)
                r = module.r['default']
                if r > 0:
                    grad_a_f32 = grad_a_f32.view(r, -1)
                preconditioned_a_grad_f32 = a_inv_f32 @ grad_a_f32
                
                # (logging for natural gradient norm)
                _total_nat_norm_sq = _total_nat_norm_sq + preconditioned_a_grad_f32.pow(2).sum() + preconditioned_b_grad_f32.pow(2).sum()

                # --- Stability: Clamp values to prevent explosions ---
                preconditioned_a_grad_f32 = torch.nan_to_num(preconditioned_a_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)
                preconditioned_b_grad_f32 = torch.nan_to_num(preconditioned_b_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)
                preconditioned_a_grad_f32 = torch.clamp(preconditioned_a_grad_f32, min=-1e6, max=1e6)
                preconditioned_b_grad_f32 = torch.clamp(preconditioned_b_grad_f32, min=-1e6, max=1e6)

                # --- OVERWRITE the original gradients ---
                lora_a.weight.grad.copy_(preconditioned_a_grad_f32.to(lora_a.weight.grad.dtype))
                lora_b.weight.grad.copy_(preconditioned_b_grad_f32.to(lora_b.weight.grad.dtype))

                # Free VRAM
                del a_inv_f32, g_inv_f32, grad_a_f32, grad_b_f32, preconditioned_a_grad_f32, preconditioned_b_grad_f32

    def neural_reprojection(self):
        """THE ADAPTATION: Prunes LoRA ranks dynamically."""
        print(f"\nGRITManager: Neural reprojection at step {self.global_step}...")
        initial_params = 0 # For logging reduction
        final_params = 0

        with torch.no_grad():
            # Calculate initial param count
            for module in self.optimized_modules:
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    initial_params += module.r['default'] * (module.in_features + module.out_features)

            for module in self.optimized_modules:
                try:
                    lora_a = module.lora_A['default']
                    lora_b = module.lora_B['default']
                    if lora_a is None or lora_b is None:
                        continue

                    A = lora_a.weight.data.float() # LoRA A matrix
                    B = lora_b.weight.data.float() # LoRA B matrix
                    r = A.shape[0] # Current rank
                    k = r # Adapted rank (to be determined)
                    M = self.a_covs[module].float() # The A-Covariance matrix

                    # --- Sanity checks ---
                    if torch.isnan(M).any() or torch.isinf(M).any():
                        final_params += r * (module.in_features + module.out_features)
                        continue
                    min_samples = int(getattr(self.config, 'kfac_min_samples', 128) or 128)
                    if self.num_samples_a.get(module, 0) < min_samples:
                        final_params += r * (module.in_features + module.out_features)
                        continue

                    # --- Determine new rank 'k' ---
                    if (self.config.enable_rank_adaptation and r > self.config.min_lora_rank and
                       self.global_step >= int(getattr(self.config, 'rank_adaptation_start_step', 0) or 0)):
                        
                        # Find the "important directions" (eigenvectors) of A-Cov
                        evals_a, V_a = torch.linalg.eigh(self.a_covs[module].float())
                        order_a = torch.argsort(evals_a, descending=True)
                        V_a = V_a[:, order_a]
                        evals_a = evals_a[order_a]
                        
                        # Find the smallest 'k' that captures X% of the "energy"
                        total_energy = torch.sum(evals_a)
                        if total_energy > 1e-6:
                            cumulative_energy = torch.cumsum(evals_a, dim=0) / total_energy
                            k = (cumulative_energy < self.config.rank_adaptation_threshold).sum().item() + 1
                        
                        k = max(k, self.config.min_lora_rank) # Enforce min rank
                        k = min(k, r) # Can't be larger than original rank
                        V_a_k = V_a[:, :k] # This is the "important subspace" (a projection matrix)

                        # Optionally, find the "important subspace" for G-Cov too
                        try:
                            g_cov_m = self.g_covs.get(module, None)
                            if getattr(self.config, 'use_two_sided_reprojection', False) and g_cov_m is not None and self.num_samples_g.get(module, 0) >= min_samples:
                                evals_g, V_g = torch.linalg.eigh(g_cov_m.float())
                                order_g = torch.argsort(evals_g, descending=True)
                                V_g_k = V_g[:, order_g][:, :k] # Subspace for G
                            else:
                                V_g_k = V_a_k # Use A's subspace for both
                        except Exception:
                            V_g_k = V_a_k
                    else:
                        # Not adapting rank, just do a simple low-rank projection
                        # (This is a form of regularization)
                        k = min(self.config.reprojection_k, r)
                        evals_a, V_a = torch.linalg.eigh(self.a_covs[module].float())
                        order_a = torch.argsort(evals_a, descending=True)
                        V_a_k = V_a[:, order_a][:, :k]
                        # ... (repeat two-sided logic) ...
                        try:
                            g_cov_m = self.g_covs.get(module, None)
                            if getattr(self.config, 'use_two_sided_reprojection', False) and g_cov_m is not None and self.num_samples_g.get(module, 0) >= min_samples:
                                evals_g, V_g = torch.linalg.eigh(g_cov_m.float())
                                order_g = torch.argsort(evals_g, descending=True)
                                V_g_k = V_g[:, order_g][:, :k]
                            else:
                                V_g_k = V_a_k
                        except Exception:
                            V_g_k = V_a_k

                    # --- Cache the subspace for the 'reproj_reg' loss term ---
                    self._last_Va_k[module] = V_a_k.detach().cpu()
                    self._last_Vg_k[module] = V_g_k.detach().cpu()

                    # --- THIS IS THE REPROJECTION ---
                    V_a_k_d = V_a_k.to(device=A.device, dtype=A.dtype, non_blocking=True)
                    V_g_k_d = V_g_k.to(device=B.device, dtype=A.dtype, non_blocking=True)
                    
                    # 1. Project A into the k-dim subspace
                    A_proj = V_a_k_d.T @ A
                    # 2. Project B into the k-dim subspace
                    B_proj = B @ V_g_k_d
                    
                    # 3. Reconstruct A and B *from* the subspace
                    # This "squeezes" the weights and throws away information
                    # that was *not* in the important subspace.
                    A_new = V_a_k_d @ A_proj
                    B_new = B_proj @ V_g_k_d.T
                    
                    # 4. OVERWRITE the LoRA weights with the new, pruned weights
                    lora_a.weight.data.copy_(A_new.to(lora_a.weight.dtype))
                    lora_b.weight.data.copy_(B_new.to(lora_b.weight.dtype))

                    # Log the new parameter count for this module
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        final_params += k * (module.in_features + module.out_features)

                except Exception as e:
                    # Don't crash training if reprojection fails on one module
                    print(f"Error during reprojection for {module.lora_name}: {e}")
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        final_params += module.r['default'] * (module.in_features + module.out_features)

        # --- Print final report ---
        if initial_params > 0:
            param_reduction = initial_params - final_params
            reduction_percent = (param_reduction / initial_params) * 100
            print(f"Neural reprojection: {initial_params:,} -> {final_params:,} ({reduction_percent:.2f}% reduction)")

        # --- Cleanup ---
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# === 4. JIT-Compiled Matrix Inverter ===
def jit_invert_tensor_pair(a_cov: torch.Tensor, g_cov: torch.Tensor, kfac_damping: float):
    """A numerically stable, JIT-compiled matrix inverter using Cholesky decomposition."""
    with torch.no_grad():
        base_damping = float(max(kfac_damping, 1e-6))
        # Ensure matrices are symmetric
        a_cov_f = a_cov.float(); a_cov_f = 0.5 * (a_cov_f + a_cov_f.T)
        g_cov_f = g_cov.float(); g_cov_f = 0.5 * (g_cov_f + g_cov_f.T)
        n_a = int(a_cov_f.shape[0]); n_g = int(g_cov_f.shape[0])
        dev = a_cov_f.device
        I_a = torch.eye(n_a, device=dev); I_g = torch.eye(n_g, device=dev)
        
        # --- Stability Trick ---
        # Try to invert with a small damping. If it fails (matrix is
        # "ill-conditioned"), try again with a slightly larger damping.
        scales = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        for s in scales:
            damp_a = float(base_damping * s)
            damp_g = float(base_damping * s)
            
            # Cholesky decomposition: A = L*L^T
            L_a, info_a = torch.linalg.cholesky_ex(a_cov_f + damp_a * I_a)
            L_g, info_g = torch.linalg.cholesky_ex(g_cov_f + damp_g * I_g)
            
            # info == 0 means it succeeded!
            if int(info_a.item()) == 0 and int(info_g.item()) == 0:
                # Invert using the Cholesky factors (fast and stable)
                a_inv = torch.cholesky_inverse(L_a).float()
                g_inv = torch.cholesky_inverse(L_g).float()
                return a_inv, g_inv
        
        # If all damping scales failed, return empty.
        return torch.empty(0, device=dev), torch.empty(0, device=dev)


# === 5. The GRIT Optimizer Wrapper ===
class GritOptimizer(Optimizer):
    """Wrapper that applies GRIT preconditioning before the underlying step."""

    def __init__(self, optimizer: Optimizer, grit_manager: 'GRITManager'):
        self.optimizer = optimizer # The "real" optimizer (e.g., AdamW)
        self.grit_manager = grit_manager

    # --- Boilerplate to pass calls through to the real optimizer ---
    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value
        
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict):
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"GritOptimizer({self.optimizer.__repr__()})"

    # --- THE HOOK ---
    def step(self, closure=None):
        """This is the function the Trainer calls."""
        # 1. Check if GRIT is ready and we haven't already preconditioned
        if self.grit_manager.factors_are_ready and not getattr(self.grit_manager, "_preconditioned", False):
            # 2. Call the manager to OVERWRITE the gradients
            self.grit_manager.precondition_gradients()
        
        # 3. Call the "real" optimizer's step(), which will now
        #    use the new, preconditioned gradients.
        self.optimizer.step(closure)


# === 6. The GRIT Callback ===
class GritCallback(TrainerCallback):
    """Connects the Trainer's 'on_step_end' event to the GRITManager's 'step'."""
    
    def __init__(self, grit_manager):
        self.grit_manager = grit_manager

    def on_step_end(self, args, state, control, **kwargs):
        """Called by the Trainer after each optimizer step."""
        # Get the most recent loss for the manager's adaptive logic
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        
        # "Tick" the GRITManager
        self.grit_manager.step(loss=last_loss)

    def on_train_end(self, args, state, control, **kwargs):
        """At the end of training, log the final ranks."""
        try:
            self.grit_manager.log_final_effective_ranks()
        except Exception:
            pass
        try:
            self.grit_manager.log_final_raw_ranks()
        except Exception:
            pass


# === 7. The GRIT Trainer ===
class GritTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that injects GRIT preconditioning and regularizers."""

    def __init__(self,processor, grit_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor # Needed for prediction_step
        self.grit_manager = grit_manager
        try:
            # (Helper for gradient clipping)
            setattr(self.grit_manager, "_last_grad_clip_cap", float(getattr(self.args, "max_grad_norm", 0.0)))
        except Exception:
            pass
        print("GritTrainer: Initialized with GRIT for Llama 3.2 model.")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Called by the Trainer to create the optimizer."""
        # 1. Create the base optimizer (e.g., AdamW)
        super().create_optimizer_and_scheduler(num_training_steps)
        
        # 2. WRAP the base optimizer with our GritOptimizer
        if self.optimizer is not None:
            print("Wrapping the optimizer with GRIT preconditioning logic.")
            self.optimizer = GritOptimizer(self.optimizer, self.grit_manager)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This is the *other* core piece of GRIT logic.
        We override compute_loss to add our two regularization penalties.
        """

        # 1. Get the model's normal loss (e.g., Cross-Entropy)
        outputs = model(**inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        # --- Get regularization strengths (lambdas) ---
        lambda_k = getattr(self.grit_manager.config, "lambda_kfac", 0.0)
        lambda_r = getattr(self.grit_manager.config, "lambda_reproj", 0.0)

        # --- Apply warmup to the penalties ---
        warmup_steps = int(getattr(self.grit_manager.config, "regularizer_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            # Linearly ramp up the penalty strength
            prog = min(1.0, max(0.0, self.grit_manager.global_step / float(warmup_steps)))
            lambda_k = float(lambda_k) * prog
            lambda_r = float(lambda_r) * prog

        # Initialize penalty terms
        curv_reg = torch.tensor(0.0, device=base_loss.device)
        reproj_reg = torch.tensor(0.0, device=base_loss.device)

        # --- Loop over all managed LoRA layers ---
        for module in getattr(self.grit_manager, "optimized_modules", []):
            lora_a = module.lora_A['default'] if 'default' in module.lora_A else None
            lora_b = module.lora_B['default'] if 'default' in module.lora_B else None
            if lora_a is None or lora_b is None:
                continue

            A = lora_a.weight # The LoRA 'A' matrix
            B = lora_b.weight # The LoRA 'B' matrix

            # Skip if weights are still on meta device
            if A.device.type == 'meta' or B.device.type == 'meta':
                continue

            a_cov = self.grit_manager.a_covs.get(module, None)
            g_cov = self.grit_manager.g_covs.get(module, None)

            # --- 1. Curvature Regularization (Fisher Penalty) ---
            # This penalty is: A^T * A_cov * A
            # It encourages the LoRA 'A' weights to align with the
            # "important" directions (eigenvectors) stored in A_cov.
            if a_cov is not None:
                A_f = A.float()
                a_cov_f = a_cov.to(A_f.device, dtype=A_f.dtype)
                curv_reg = curv_reg + ((a_cov_f @ A_f) * A_f).sum()

            # Same penalty for 'B' weights and G_cov.
            if g_cov is not None:
                B_f = B.float()
                g_cov_f = g_cov.to(B_f.device, dtype=B_f.dtype)
                curv_reg = curv_reg + ((B_f @ g_cov_f) * B_f).sum()

            # --- 2. Reprojection Regularization ---
            # Get the "important subspace" (V_a_k) that was
            # calculated during the `neural_reprojection` step.
            Va_cache = getattr(self.grit_manager, "_last_Va_k", {})
            Vg_cache = getattr(self.grit_manager, "_last_Vg_k", {})
            V_a_k = Va_cache.get(module, None)
            V_g_k = Vg_cache.get(module, None)

            # (If it's not cached, try to calculate it on the fly)
            if V_a_k is None and a_cov is not None:
                # ... (omitted: complex on-the-fly calculation of V_a_k) ...
                # This logic is a fallback in case reprojection hasn't run yet
                pass # The main logic is in the 'else' block below
            
            # If the subspace (V_a_k) is ready:
            if V_a_k is not None and V_a_k.numel() > 0:
                A_f = A.float(); B_f = B.float()
                device, dtype = A_f.device, A_f.dtype
                
                # Load the subspace matrices to the GPU
                V_a_k_d = V_a_k.to(device=device, dtype=dtype, non_blocking=True)
                V_g_k_d = (V_g_k if V_g_k is not None else V_a_k).to(device=device, dtype=dtype, non_blocking=True)
                
                # P = V * V^T is the projection matrix onto the subspace
                P_a = V_a_k_d @ V_a_k_d.T
                I_a = torch.eye(P_a.shape[0], device=device, dtype=dtype)
                P_g = V_g_k_d @ V_g_k_d.T
                I_g = torch.eye(P_g.shape[0], device=device, dtype=dtype)
                
                # A_res = (I - P) * A
                # This calculates the part of the 'A' matrix that is *NOT*
                # in the "important subspace" (the part "sticking out").
                A_res = (I_a - P_a) @ A_f
                B_res = B_f @ (I_g - P_g)
                
                # We PENALIZE this "residual" part. This forces the
                # weights to stay "squeezed" into the low-rank subspace.
                reproj_reg = reproj_reg + (A_res.pow(2).sum() + B_res.pow(2).sum())

        # --- 3. Combine losses ---
        loss = base_loss + lambda_k * curv_reg + lambda_r * reproj_reg
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Standard training step, but it calls our custom compute_loss."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use our custom `compute_loss`
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs) 
            
        if self.args.n_gpu > 1:
            loss = loss.mean()
            
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def optimizer_step(self, *args, **kwargs):
        """A final, failsafe hook to ensure preconditioning happens."""
        setattr(self.grit_manager, "_preconditioned", False)
        if getattr(self.grit_manager, "factors_are_ready", False):
            self.grit_manager.precondition_gradients()
            setattr(self.grit_manager, "_preconditioned", True)
        
        # Call the base Trainer's optimizer step, which
        # will call our GritOptimizer's `step()` method.
        return super().optimizer_step(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Add VRAM clearing before evaluation."""
        print("\nClearing VRAM before evaluation...")
        gc.collect()
        torch.cuda.empty_cache()
        return super().evaluate(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Custom logic for VQA-style evaluation (using .generate)."""
        
        # --- Standard setup ---
        has_labels = "labels" in inputs
        if has_labels:
            labels = inputs.pop("labels") # Pop labels, as .generate() doesn't want them
        else:
            labels = None

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]  # ***CRITICAL: Save the prompt length***

        with torch.no_grad():
            if prediction_loss_only:
                # ... (standard loss calculation) ...
                outputs = model(**inputs)
                loss = outputs.loss if has_labels else None
                return (loss, None, None)

            # --- Call model.generate() to get the text answer ---
            gen_kwargs = {
                k: v for k, v in inputs.items() 
                if k in [
                    "pixel_values", 
                    "aspect_ratio_ids", 
                    "aspect_ratio_mask", 
                    "cross_attention_mask"
                ]
            }

            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,  # Short answer for VQA # before value 32
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                # Pass vision inputs (pixel_values, etc.) to .generate()
                # Llama 3.2 Vision needs 'image_sizes' to be passed
                # **{k: v for k, v in inputs.items() if k in ["pixel_values", "image_sizes"]}
                **gen_kwargs,
            )

            # *** CRITICAL VQA FIX ***
            # The `generated_tokens` contains the *entire* sequence: (PROMPT + ANSWER)
            # We must SLICE off the prompt to get *only* the new answer tokens.
            generated_tokens = generated_tokens[:, input_len:]

            # --- Calculate the loss (optional) ---
            loss = None
            if has_labels:
                inputs_copy = inputs.copy() # Get original inputs
                inputs_copy["labels"] = labels # Put labels back
                loss = self.compute_loss(model, inputs_copy).detach()

        # (Label preparation, standard)
        if labels is not None:
            if labels.dim() == 3:
                labels = torch.argmax(labels, dim=-1)
            elif labels.dim() == 1:
                batch_size = input_ids.shape[0]
                labels = labels.view(batch_size, -1)

        # Return (loss, predictions, labels)
        # `generated_tokens` is now *just* the answer, which matches
        # what the evaluation metrics (ROUGE, etc.) expect.
        return loss, generated_tokens, labels