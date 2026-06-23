import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================
class CropDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        """Load and preprocess crop data"""
        self.data = pd.read_csv(csv_file)
        
        # Convert dates to days since planting
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['planting_date'] = pd.to_datetime(self.data['planting_date'])
        self.data['days_since_planting'] = (
            self.data['date'] - self.data['planting_date']
        ).dt.days
        
        # Encode categorical variables
        self.data['soil_texture_encoded'] = pd.Categorical(
            self.data['soil_texture']
        ).codes
        self.data['variety_encoded'] = pd.Categorical(
            self.data['variety']
        ).codes
        
        # Define feature columns
        self.weather_features = ['tmax', 'tmin', 'rainfall', 'solar_radiation']
        self.soil_features = ['soil_pH', 'soil_N', 'soil_texture_encoded']
        self.management_features = ['fertilizer_N', 'irrigation', 'variety_encoded']
        self.time_features = ['days_since_planting']
        
        # Target variables
        self.targets = ['LAI', 'biomass', 'soil_water']
        
        # Normalize features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        all_features = (self.weather_features + self.soil_features + 
                       self.management_features + self.time_features)
        
        self.X = self.scaler_X.fit_transform(self.data[all_features])
        self.y = self.scaler_y.fit_transform(self.data[self.targets])
        
        # Store raw weather data for physics calculations
        self.weather_raw = self.data[self.weather_features].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'X': torch.FloatTensor(self.X[idx]),
            'y': torch.FloatTensor(self.y[idx]),
            'weather': torch.FloatTensor(self.weather_raw[idx]),
            'time': torch.FloatTensor([self.data.iloc[idx]['days_since_planting']])
        }

# ============================================
# 2. ENHANCED PINN MODEL ARCHITECTURE
# ============================================
class EnhancedCropPINN(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=[256, 128, 64], output_dim=3):
        super(EnhancedCropPINN, self).__init__()
        
        # Neural Network with residual connections
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # Skip connection
        self.skip = nn.Linear(input_dim, hidden_dims[2])
        
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
        # Physics parameters with reasonable initial values
        self.RUE = nn.Parameter(torch.tensor(3.0))  # Radiation Use Efficiency (g/MJ)
        self.k = nn.Parameter(torch.tensor(0.6))     # Light extinction coefficient
        self.Kc = nn.Parameter(torch.tensor(1.1))    # Crop coefficient for ET
        self.SLA = nn.Parameter(torch.tensor(0.025)) # Specific Leaf Area (m²/g)
        
        # Temperature parameters
        self.T_base = nn.Parameter(torch.tensor(10.0))  # Base temperature (°C)
        self.T_opt = nn.Parameter(torch.tensor(25.0))   # Optimal temperature (°C)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        # Main path - use eval mode for BatchNorm when batch size is 1
        if x.size(0) == 1:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
        
        h1 = self.dropout(self.activation(self.bn1(self.layer1(x))))
        h2 = self.dropout(self.activation(self.bn2(self.layer2(h1))))
        h3 = self.activation(self.bn3(self.layer3(h2)))
        
        # Skip connection
        skip = self.skip(x)
        h3 = h3 + skip
        
        output = self.output_layer(h3)
        
        # Soft constraints using sigmoid for bounded outputs
        LAI = 8.0 * torch.sigmoid(output[:, 0])          # Max LAI = 8
        biomass = 20000.0 * torch.sigmoid(output[:, 1])  # Max biomass = 20000 kg/ha
        soil_water = 400.0 * torch.sigmoid(output[:, 2]) # Max SW = 400 mm
        
        return torch.stack([LAI, biomass, soil_water], dim=1)
    
    def temperature_stress(self, temp_avg):
        """
        Calculate temperature stress factor (0-1)
        Uses cardinal temperature model
        """
        T_base = torch.clamp(self.T_base, 5.0, 15.0)
        T_opt = torch.clamp(self.T_opt, 20.0, 30.0)
        T_max = 40.0
        
        stress = torch.zeros_like(temp_avg)
        
        # Below base temperature: no growth
        mask1 = temp_avg < T_base
        stress[mask1] = 0.0
        
        # Between base and optimal: linear increase
        mask2 = (temp_avg >= T_base) & (temp_avg <= T_opt)
        stress[mask2] = (temp_avg[mask2] - T_base) / (T_opt - T_base)
        
        # Between optimal and maximum: linear decrease
        mask3 = (temp_avg > T_opt) & (temp_avg <= T_max)
        stress[mask3] = (T_max - temp_avg[mask3]) / (T_max - T_opt)
        
        # Above maximum: no growth
        mask4 = temp_avg > T_max
        stress[mask4] = 0.0
        
        return stress
    
    def enhanced_physics_loss(self, predictions, weather, time_steps):
        """
        Calculate enhanced physics-based loss to enforce biological constraints
        """
        LAI = predictions[:, 0]
        biomass = predictions[:, 1]
        soil_water = predictions[:, 2]
        
        # Extract weather variables
        solar_radiation = weather[:, 3]
        rainfall = weather[:, 2]
        temp_max = weather[:, 0]
        temp_min = weather[:, 1]
        temp_avg = (temp_max + temp_min) / 2
        
        # Clamp physics parameters to reasonable ranges
        RUE_clamped = torch.clamp(self.RUE, 1.0, 5.0)
        k_clamped = torch.clamp(self.k, 0.3, 1.0)
        Kc_clamped = torch.clamp(self.Kc, 0.8, 1.5)
        SLA_clamped = torch.clamp(self.SLA, 0.015, 0.035)
        
        # Calculate temperature stress factor
        temp_stress = self.temperature_stress(temp_avg)
        
        # ==========================================
        # PHYSICS EQUATION 1: BIOMASS GROWTH
        # dW/dt = RUE × IPAR × f(T)
        # ==========================================
        
        # PAR is approximately 50% of solar radiation (MJ/m²/day)
        PAR = solar_radiation * 0.5
        
        # Intercepted PAR using Beer's Law
        IPAR = PAR * (1 - torch.exp(-k_clamped * LAI))
        
        # Growth potential with temperature stress
        growth_potential = RUE_clamped * IPAR * temp_stress
        
        physics_loss_growth = torch.tensor(0.0, device=predictions.device)
        if len(biomass) > 1:
            # Compute actual growth rate from predictions
            biomass_diff = biomass[1:] - biomass[:-1]
            # Ensure positive growth (can't lose biomass except senescence)
            biomass_diff = torch.clamp(biomass_diff, min=0)
            
            # Convert units: growth_potential is in g/m²/day, biomass is in kg/ha
            # 1 g/m²/day = 10 kg/ha/day
            growth_potential_scaled = growth_potential[:-1] * 10.0
            
            physics_loss_growth = torch.mean(
                (biomass_diff - growth_potential_scaled)**2
            )
        
        # ==========================================
        # PHYSICS EQUATION 2: LAI-BIOMASS RELATIONSHIP
        # LAI = Biomass × SLA × Partition_Coefficient
        # ==========================================
        
        # Partition coefficient: fraction of biomass in leaves
        # Varies with growth stage (higher early, lower at maturity)
        time_normalized = torch.clamp(time_steps.squeeze() / 100.0, 0, 1)
        partition_coef = 0.6 - 0.3 * time_normalized  # 0.6 → 0.3
        
        # Convert biomass from kg/ha to g/m² for consistency
        # 1 kg/ha = 0.1 g/m²
        biomass_g_m2 = biomass * 0.1
        
        LAI_from_biomass = biomass_g_m2 * SLA_clamped * partition_coef
        
        physics_loss_lai_biomass = torch.mean((LAI - LAI_from_biomass)**2)
        
        # ==========================================
        # PHYSICS EQUATION 3: WATER BALANCE
        # dSW/dt = P + I - ET - RO - D
        # ==========================================
        
        # Reference evapotranspiration (Hargreaves equation)
        temp_range = torch.clamp(temp_max - temp_min, min=0.1)
        ET0 = 0.0023 * (temp_avg + 17.8) * torch.sqrt(temp_range) * solar_radiation * 0.408
        
        # Actual ET with crop coefficient and water stress
        water_stress = torch.clamp(soil_water / 200.0, 0.0, 1.0)
        ET_actual = Kc_clamped * ET0 * water_stress
        
        # Runoff (simple threshold model)
        field_capacity = 250.0
        storage_available = torch.clamp(field_capacity - soil_water, min=0)
        runoff = torch.relu(rainfall - storage_available)
        
        # Deep drainage (percolation)
        drainage = 0.05 * torch.relu(soil_water - field_capacity)
        
        # Water balance
        water_balance_theory = rainfall - ET_actual - runoff - drainage
        
        physics_loss_water = torch.tensor(0.0, device=predictions.device)
        if len(soil_water) > 1:
            water_diff = soil_water[1:] - soil_water[:-1]
            physics_loss_water = torch.mean(
                (water_diff - water_balance_theory[:-1])**2
            )
        
        # ==========================================
        # PHYSICS EQUATION 4: MONOTONICITY CONSTRAINTS
        # Biomass should generally increase over time
        # ==========================================
        
        physics_loss_monotonic = torch.tensor(0.0, device=predictions.device)
        if len(biomass) > 1:
            # Penalize biomass decrease
            biomass_decrease = torch.relu(biomass[:-1] - biomass[1:])
            physics_loss_monotonic = torch.mean(biomass_decrease**2)
        
        # ==========================================
        # PHYSICS EQUATION 5: SENESCENCE
        # LAI should decrease after maturity
        # ==========================================
        
        physics_loss_senescence = torch.tensor(0.0, device=predictions.device)
        maturity_days = 90  # Days to maturity
        
        late_season_mask = time_steps.squeeze() > maturity_days
        if torch.any(late_season_mask) and len(LAI) > 1:
            # Find indices where both current and next are in late season
            late_indices = torch.where(late_season_mask)[0]
            if len(late_indices) > 1:
                # LAI should not increase significantly after maturity
                for i in range(len(late_indices) - 1):
                    idx = late_indices[i]
                    if idx + 1 < len(LAI):
                        LAI_increase = torch.relu(LAI[idx + 1] - LAI[idx])
                        physics_loss_senescence += LAI_increase**2
                
                if len(late_indices) > 1:
                    physics_loss_senescence /= (len(late_indices) - 1)
        
        # ==========================================
        # PHYSICS EQUATION 6: ENERGY CONSERVATION
        # Total energy captured should be consistent
        # ==========================================
        
        # Cumulative intercepted radiation
        cumulative_IPAR = torch.cumsum(IPAR, dim=0)
        # Cumulative biomass growth
        cumulative_biomass = biomass
        
        # Energy use efficiency should be relatively constant
        physics_loss_energy = torch.tensor(0.0, device=predictions.device)
        if len(cumulative_biomass) > 10:  # Need enough samples
            # Avoid division by zero
            valid_mask = cumulative_IPAR > 1.0
            if torch.any(valid_mask):
                efficiency = cumulative_biomass[valid_mask] / (cumulative_IPAR[valid_mask] + 1e-6)
                # Efficiency should be relatively stable (low variance)
                physics_loss_energy = torch.var(efficiency)
        
        # ==========================================
        # COMBINE ALL PHYSICS LOSSES
        # ==========================================
        
        total_physics_loss = (
            1.0 * physics_loss_growth +
            0.5 * physics_loss_water +
            0.3 * physics_loss_lai_biomass +
            0.2 * physics_loss_monotonic +
            0.1 * physics_loss_senescence +
            0.05 * physics_loss_energy
        )
        
        components = {
            'growth': physics_loss_growth.item() if isinstance(physics_loss_growth, torch.Tensor) else 0,
            'water': physics_loss_water.item() if isinstance(physics_loss_water, torch.Tensor) else 0,
            'lai_biomass': physics_loss_lai_biomass.item() if isinstance(physics_loss_lai_biomass, torch.Tensor) else 0,
            'monotonic': physics_loss_monotonic.item() if isinstance(physics_loss_monotonic, torch.Tensor) else 0,
            'senescence': physics_loss_senescence.item() if isinstance(physics_loss_senescence, torch.Tensor) else 0,
            'energy': physics_loss_energy.item() if isinstance(physics_loss_energy, torch.Tensor) else 0,
        }
        
        return total_physics_loss, components
    
    def boundary_loss(self, predictions, time_steps):
        """
        Enforce boundary conditions
        """
        loss = torch.tensor(0.0, device=predictions.device)
        
        # Convert [batch_size, 1] -> [batch_size]
        time_steps = time_steps.squeeze()
        
        # Initial growth stage (first 5 days)
        initial_mask = time_steps < 5
        
        if torch.any(initial_mask):
            LAI_initial = predictions[initial_mask, 0]
            biomass_initial = predictions[initial_mask, 1]
            
            # LAI should start near zero
            loss += torch.mean(LAI_initial ** 2)
            
            # Biomass should start small (around 50 kg/ha)
            loss += torch.mean((biomass_initial - 50) ** 2) / 1000
        
        # Physical limits (soft constraints - already handled in forward pass)
        # These are additional penalties if values go beyond sigmoid bounds
        
        LAI = predictions[:, 0]
        biomass = predictions[:, 1]
        soil_water = predictions[:, 2]
        
        # Additional upper bound penalties
        LAI_max = 8.0
        loss += torch.mean(torch.relu(LAI - LAI_max) ** 2)
        
        biomass_max = 20000
        loss += torch.mean(torch.relu(biomass - biomass_max) ** 2) / 1e6
        
        SW_max = 400
        loss += torch.mean(torch.relu(soil_water - SW_max) ** 2) / 1000
        
        # Lower bound penalties (should be handled by ReLU, but add for safety)
        loss += torch.mean(torch.relu(-LAI) ** 2)
        loss += torch.mean(torch.relu(-biomass) ** 2) / 1e6
        loss += torch.mean(torch.relu(-soil_water) ** 2) / 1000
        
        return loss

# ============================================
# 3. ENHANCED TRAINING FUNCTION
# ============================================
def train_pinn_enhanced(model, train_loader, val_loader, epochs=1000, device='cpu'):
    """
    Train the enhanced PINN model with curriculum learning
    """
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'physics_loss': [], 'boundary_loss': [],
        'learning_rate': [],
        'components': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 50
    
    print(f"Training on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("=" * 80)
    
    for epoch in range(epochs):
        # Curriculum learning: gradually increase physics loss weight
        # Start with mostly data-driven, gradually add physics
        lambda_physics = min(0.5, 0.01 + (epoch / epochs) * 0.49)
        lambda_boundary = 0.05
        
        # ================== Training ==================
        model.train()
        train_loss_total = 0
        data_loss_total = 0
        physics_loss_total = 0
        boundary_loss_total = 0
        
        components_epoch = {
            'growth': 0, 'water': 0, 'lai_biomass': 0,
            'monotonic': 0, 'senescence': 0, 'energy': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            X = batch['X'].to(device)
            y_true = batch['y'].to(device)
            weather = batch['weather'].to(device)
            time = batch['time'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X)
            
            # Data loss with per-output weighting
            # LAI is most important, biomass scaled down due to magnitude
            weights = torch.tensor([1.0, 0.1, 0.5], device=device)
            data_loss = torch.mean(weights * torch.mean((y_pred - y_true)**2, dim=0))
            
            # Physics loss
            physics_loss, components = model.enhanced_physics_loss(y_pred, weather, time)
            
            # Boundary loss
            boundary_loss = model.boundary_loss(y_pred, time)
            
            # Total loss
            total_loss = (
                data_loss +
                lambda_physics * physics_loss +
                lambda_boundary * boundary_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_loss_total += total_loss.item()
            data_loss_total += data_loss.item()
            physics_loss_total += physics_loss.item()
            boundary_loss_total += boundary_loss.item()
            
            for key in components:
                components_epoch[key] += components[key]
        
        # Average component losses
        for key in components_epoch:
            components_epoch[key] /= len(train_loader)
        
        # ================== Validation ==================
        model.eval()
        val_loss_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y_true = batch['y'].to(device)
                weather = batch['weather'].to(device)
                time = batch['time'].to(device)
                
                y_pred = model(X)
                
                weights = torch.tensor([1.0, 0.1, 0.5], device=device)
                data_loss = torch.mean(weights * torch.mean((y_pred - y_true)**2, dim=0))
                physics_loss, components = model.enhanced_physics_loss(y_pred, weather, time)
                boundary_loss = model.boundary_loss(y_pred, time)
                
                total_loss = (
                    data_loss +
                    lambda_physics * physics_loss +
                    lambda_boundary * boundary_loss
                )
                
                val_loss_total += total_loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        avg_physics_loss = physics_loss_total / len(train_loader)
        avg_boundary_loss = boundary_loss_total / len(train_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['data_loss'].append(avg_data_loss)
        history['physics_loss'].append(avg_physics_loss)
        history['boundary_loss'].append(avg_boundary_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['components'].append(components_epoch.copy())
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'history': history,
            }, 'best_pinn_model_enhanced.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}] - λ_physics={lambda_physics:.3f}")
            print(f"  Losses - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            print(f"  Components:")
            print(f"    Data: {avg_data_loss:.6f}")
            print(f"    Physics: {avg_physics_loss:.6f} (Growth: {components_epoch['growth']:.6f}, "
                  f"Water: {components_epoch['water']:.6f}, LAI-Bio: {components_epoch['lai_biomass']:.6f})")
            print(f"    Boundary: {avg_boundary_loss:.6f}")
            print(f"  Learned Parameters:")
            print(f"    RUE: {model.RUE.item():.3f} g/MJ | k: {model.k.item():.3f} | "
                  f"Kc: {model.Kc.item():.3f} | SLA: {model.SLA.item():.4f} m²/g")
            print(f"    T_base: {model.T_base.item():.1f}°C | T_opt: {model.T_opt.item():.1f}°C")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 80)
    
    return history

# ============================================
# 4. EVALUATION FUNCTIONS
# ============================================
def evaluate_model(model, test_loader, dataset, device='cpu'):
    """Evaluate model performance on test set"""
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y_true = batch['y']
            
            y_pred = model(X).cpu()
            
            predictions.append(y_pred.numpy())
            actuals.append(y_true.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverse transform to original scale
    predictions = dataset.scaler_y.inverse_transform(predictions)
    actuals = dataset.scaler_y.inverse_transform(actuals)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals), axis=0)
    rmse = np.sqrt(np.mean((predictions - actuals)**2, axis=0))
    
    # R² score
    r2 = []
    for i in range(3):
        ss_res = np.sum((actuals[:, i] - predictions[:, i])**2)
        ss_tot = np.sum((actuals[:, i] - np.mean(actuals[:, i]))**2)
        r2.append(1 - (ss_res / ss_tot))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = []
    for i in range(3):
        # Avoid division by zero
        mask = actuals[:, i] > 0.1
        if np.any(mask):
            mape.append(np.mean(np.abs((actuals[mask, i] - predictions[mask, i]) / actuals[mask, i])) * 100)
        else:
            mape.append(0)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    print(f"\n{'Metric':<15} {'LAI':<15} {'Biomass (kg/ha)':<20} {'Soil Water (mm)':<20}")
    print("-"*80)
    print(f"{'MAE':<15} {mae[0]:<15.3f} {mae[1]:<20.1f} {mae[2]:<20.1f}")
    print(f"{'RMSE':<15} {rmse[0]:<15.3f} {rmse[1]:<20.1f} {rmse[2]:<20.1f}")
    print(f"{'R²':<15} {r2[0]:<15.3f} {r2[1]:<20.3f} {r2[2]:<20.3f}")
    print(f"{'MAPE (%)':<15} {mape[0]:<15.1f} {mape[1]:<20.1f} {mape[2]:<20.1f}")
    print("="*80)
    
    return predictions, actuals, {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

def predict_with_uncertainty(model, test_loader, dataset, device='cpu', n_samples=50):
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    model = model.to(device)
    model.train()  # Keep dropout active for MC sampling
    
    predictions_samples = []
    actuals = []
    
    print(f"\nGenerating {n_samples} Monte Carlo samples for uncertainty estimation...")
    
    for sample in range(n_samples):
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                X = batch['X'].to(device)
                if sample == 0:  # Only collect actuals once
                    actuals.append(batch['y'].numpy())
                
                y_pred = model(X).cpu()
                predictions.append(y_pred.numpy())
        
        predictions_samples.append(np.vstack(predictions))
        
        if (sample + 1) % 10 == 0:
            print(f"  Completed {sample + 1}/{n_samples} samples")
    
    predictions_samples = np.array(predictions_samples)  # [n_samples, n_test, 3]
    actuals = np.vstack(actuals)
    
    # Calculate mean and std across MC samples
    mean_pred = np.mean(predictions_samples, axis=0)
    std_pred = np.std(predictions_samples, axis=0)
    
    # Inverse transform to original scale
    mean_pred = dataset.scaler_y.inverse_transform(mean_pred)
    actuals = dataset.scaler_y.inverse_transform(actuals)
    
    # Transform std (approximate scaling)
    std_pred = std_pred * dataset.scaler_y.scale_
    
    return mean_pred, std_pred, actuals

# ============================================
# 5. VISUALIZATION FUNCTIONS
# ============================================
def plot_training_history(history):
    """Plot comprehensive training history"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Loss Components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['data_loss'], label='Data Loss', linewidth=2)
    ax2.plot(history['physics_loss'], label='Physics Loss', linewidth=2)
    ax2.plot(history['boundary_loss'], label='Boundary Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['learning_rate'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Physics Components - Growth
    ax4 = fig.add_subplot(gs[1, 0])
    growth_losses = [comp['growth'] for comp in history['components']]
    ax4.plot(growth_losses, linewidth=2, color='brown')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Growth Physics Loss', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Physics Components - Water
    ax5 = fig.add_subplot(gs[1, 1])
    water_losses = [comp['water'] for comp in history['components']]
    ax5.plot(water_losses, linewidth=2, color='blue')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Water Balance Physics Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Physics Components - LAI-Biomass
    ax6 = fig.add_subplot(gs[1, 2])
    lai_bio_losses = [comp['lai_biomass'] for comp in history['components']]
    ax6.plot(lai_bio_losses, linewidth=2, color='green')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title('LAI-Biomass Physics Loss', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Physics Components - Monotonic
    ax7 = fig.add_subplot(gs[2, 0])
    monotonic_losses = [comp['monotonic'] for comp in history['components']]
    ax7.plot(monotonic_losses, linewidth=2, color='purple')
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Loss', fontsize=12)
    ax7.set_title('Monotonicity Physics Loss', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Physics Components - Senescence
    ax8 = fig.add_subplot(gs[2, 1])
    senescence_losses = [comp['senescence'] for comp in history['components']]
    ax8.plot(senescence_losses, linewidth=2, color='orange')
    ax8.set_xlabel('Epoch', fontsize=12)
    ax8.set_ylabel('Loss', fontsize=12)
    ax8.set_title('Senescence Physics Loss', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Physics Components - Energy
    ax9 = fig.add_subplot(gs[2, 2])
    energy_losses = [comp['energy'] for comp in history['components']]
    ax9.plot(energy_losses, linewidth=2, color='red')
    ax9.set_xlabel('Epoch', fontsize=12)
    ax9.set_ylabel('Loss', fontsize=12)
    ax9.set_title('Energy Conservation Physics Loss', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Enhanced PINN', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(predictions, actuals, metrics):
    """Plot prediction vs actual scatter plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)']
    colors = ['green', 'brown', 'blue']
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(actuals[:, i], predictions[:, i], alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(actuals[:, i].min(), predictions[:, i].min())
        max_val = max(actuals[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics text
        textstr = f'R² = {metrics["r2"][i]:.3f}\nRMSE = {metrics["rmse"][i]:.2f}\nMAE = {metrics["mae"][i]:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel(f'Actual {var}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted {var}', fontsize=12, fontweight='bold')
        ax.set_title(f'{var} Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty(mean_pred, std_pred, actuals, sample_indices=None):
    """Plot predictions with uncertainty bands"""
    if sample_indices is None:
        sample_indices = range(min(200, len(actuals)))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)']
    colors = ['green', 'brown', 'blue']
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i]
        
        x = np.array(sample_indices)
        
        # Plot actual values
        ax.plot(x, actuals[sample_indices, i], 'o', color='black', 
                label='Actual', markersize=4, alpha=0.6)
        
        # Plot mean predictions
        ax.plot(x, mean_pred[sample_indices, i], '-', color=color, 
                linewidth=2, label='Predicted Mean')
        
        # Plot uncertainty bands (±2 std = ~95% confidence interval)
        ax.fill_between(x,
                        mean_pred[sample_indices, i] - 2*std_pred[sample_indices, i],
                        mean_pred[sample_indices, i] + 2*std_pred[sample_indices, i],
                        alpha=0.3, color=color, label='95% Confidence Interval')
        
        ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax.set_ylabel(var, fontsize=12, fontweight='bold')
        ax.set_title(f'{var} Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(predictions, actuals):
    """Plot residual analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)']
    colors = ['green', 'brown', 'blue']
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        residuals = actuals[:, i] - predictions[:, i]
        
        # Residuals vs Predicted
        ax1 = axes[0, i]
        ax1.scatter(predictions[:, i], residuals, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel(f'Predicted {var}', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax1.set_title(f'{var} Residuals', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residuals Distribution
        ax2 = axes[1, i]
        ax2.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'{var} Residuals Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        textstr = f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series(predictions, actuals, dataset, test_dataset, num_samples=3):
    """
    Plot time series for selected samples
    """
    # Get the actual indices used in the test set
    test_indices = test_dataset.indices
    
    # Get time information for test samples only
    time_data = dataset.data['days_since_planting'].values[test_indices]
    
    # Get plot IDs if available
    if 'plot_id' in dataset.data.columns:
        plot_ids = dataset.data['plot_id'].values[test_indices]
        unique_ids = np.unique(plot_ids)
    else:
        unique_ids = None
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)']
    colors = ['green', 'brown', 'blue']
    
    if unique_ids is not None and len(unique_ids) >= num_samples:
        # Plot by plot_id
        selected_ids = np.random.choice(unique_ids, min(num_samples, len(unique_ids)), replace=False)
        
        for row, plot_id in enumerate(selected_ids):
            # Get indices for this plot within test set
            mask = plot_ids == plot_id
            local_indices = np.where(mask)[0]
            
            if len(local_indices) == 0:
                continue
            
            # Sort by time for proper line plotting
            time_subset = time_data[local_indices]
            sort_idx = np.argsort(time_subset)
            local_indices_sorted = local_indices[sort_idx]
            time_subset_sorted = time_subset[sort_idx]
            
            for col, (var, color) in enumerate(zip(variables, colors)):
                ax = axes[row, col]
                
                # Plot actual and predicted
                ax.plot(time_subset_sorted, actuals[local_indices_sorted, col], 'o-', 
                       color='black', label='Actual', markersize=5, linewidth=2)
                ax.plot(time_subset_sorted, predictions[local_indices_sorted, col], 's-', 
                       color=color, label='Predicted', markersize=5, linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Days Since Planting', fontsize=11, fontweight='bold')
                ax.set_ylabel(var, fontsize=11, fontweight='bold')
                ax.set_title(f'{var} - Plot {plot_id}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
    else:
        # No plot_id available, take sequential chunks
        samples_per_plot = max(1, len(predictions) // num_samples)
        
        for row in range(num_samples):
            start_idx = row * samples_per_plot
            end_idx = min(start_idx + samples_per_plot, len(predictions))
            
            if start_idx >= len(predictions):
                break
                
            indices = np.arange(start_idx, end_idx)
            
            # Sort by time
            time_subset = time_data[indices]
            sort_idx = np.argsort(time_subset)
            indices_sorted = indices[sort_idx]
            time_subset_sorted = time_subset[sort_idx]
            
            for col, (var, color) in enumerate(zip(variables, colors)):
                ax = axes[row, col]
                
                # Plot actual and predicted
                ax.plot(time_subset_sorted, actuals[indices_sorted, col], 'o-', 
                       color='black', label='Actual', markersize=4, linewidth=2)
                ax.plot(time_subset_sorted, predictions[indices_sorted, col], 's-', 
                       color=color, label='Predicted', markersize=4, linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Days Since Planting', fontsize=11, fontweight='bold')
                ax.set_ylabel(var, fontsize=11, fontweight='bold')
                ax.set_title(f'{var} - Subset {row+1}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Saved: time_series_predictions.png")

def plot_learned_parameters(model):
    """Visualize learned physics parameters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Get parameter values
    params = {
        'RUE': model.RUE.item(),
        'k': model.k.item(),
        'Kc': model.Kc.item(),
        'SLA': model.SLA.item(),
        'T_base': model.T_base.item(),
        'T_opt': model.T_opt.item(),
    }
    
    # Expected ranges (from literature)
    expected_ranges = {
        'RUE': (1.5, 4.5, 'g/MJ'),
        'k': (0.4, 0.8, ''),
        'Kc': (0.9, 1.3, ''),
        'SLA': (0.018, 0.030, 'm²/g'),
        'T_base': (8, 12, '°C'),
        'T_opt': (23, 28, '°C'),
    }
    
    param_names = list(params.keys())
    
    for idx, (param_name, ax) in enumerate(zip(param_names, axes.flat)):
        value = params[param_name]
        exp_min, exp_max, unit = expected_ranges[param_name]
        
        # Create bar showing learned value within expected range
        ax.barh(0, value, height=0.5, color='green', alpha=0.7, label='Learned Value')
        ax.axvline(x=exp_min, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Expected Range')
        ax.axvline(x=exp_max, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        # Shade expected range
        ax.axvspan(exp_min, exp_max, alpha=0.2, color='yellow')
        
        ax.set_xlim(exp_min - 0.2*(exp_max-exp_min), exp_max + 0.2*(exp_max-exp_min))
        ax.set_yticks([])
        ax.set_xlabel(f'{param_name} ({unit})' if unit else param_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{param_name} = {value:.3f} {unit}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Learned Physics Parameters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learned_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# 6. SYNTHETIC DATA GENERATION (FOR TESTING)
# ============================================
def generate_synthetic_crop_data(n_samples=5000, output_file='crop_data.csv'):
    """
    Generate synthetic crop data for testing the PINN
    """
    print("Generating synthetic crop data...")
    
    np.random.seed(42)
    
    # Number of plots and days
    n_plots = 50
    days_per_plot = n_samples // n_plots
    
    data = []
    
    for plot_id in range(n_plots):
        # Random planting date
        planting_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        # Random variety and soil
        variety = np.random.choice(['Variety_A', 'Variety_B', 'Variety_C'])
        soil_texture = np.random.choice(['Clay', 'Loam', 'Sandy'])
        soil_pH = np.random.uniform(5.5, 7.5)
        soil_N = np.random.uniform(50, 150)
        
        # Initial conditions
        LAI = 0.1
        biomass = 50.0
        soil_water = 200.0
        
        # Physics parameters (with some variation per plot)
        RUE = np.random.uniform(2.5, 3.5)
        k = np.random.uniform(0.5, 0.7)
        Kc = np.random.uniform(1.0, 1.2)
        SLA = np.random.uniform(0.020, 0.028)
        
        for day in range(days_per_plot):
            date = planting_date + pd.Timedelta(days=day)
            
            # Generate weather (seasonal pattern)
            doy = date.dayofyear
            tmax = 25 + 5 * np.sin(2*np.pi*doy/365) + np.random.normal(0, 2)
            tmin = tmax - 10 + np.random.normal(0, 1)
            solar_radiation = 18 + 4 * np.sin(2*np.pi*doy/365) + np.random.normal(0, 1)
            rainfall = max(0, np.random.gamma(2, 2))
            
            # Management
            fertilizer_N = 100 if day == 30 else 0
            irrigation = 20 if day % 7 == 0 and rainfall < 5 else 0
            
            # Simulate crop growth
            temp_avg = (tmax + tmin) / 2
            
            # Temperature stress
            T_base = 10
            T_opt = 25
            T_max = 40
            if temp_avg < T_base:
                temp_stress = 0
            elif temp_avg < T_opt:
                temp_stress = (temp_avg - T_base) / (T_opt - T_base)
            elif temp_avg < T_max:
                temp_stress = (T_max - temp_avg) / (T_max - T_opt)
            else:
                temp_stress = 0
            
            # Growth
            PAR = solar_radiation * 0.5
            IPAR = PAR * (1 - np.exp(-k * LAI))
            daily_growth = RUE * IPAR * temp_stress * 10  # kg/ha/day
            
            # LAI growth
            partition = 0.6 - 0.3 * (day / days_per_plot)
            LAI_growth = daily_growth * 0.1 * SLA * partition
            
            # Senescence after 90 days
            if day > 90:
                LAI_growth -= 0.05 * LAI
            
            LAI = max(0, min(8, LAI + LAI_growth))
            biomass = max(50, min(20000, biomass + daily_growth))
            
            # Water balance
            ET0 = 0.0023 * (temp_avg + 17.8) * np.sqrt(max(0.1, tmax - tmin)) * solar_radiation * 0.408
            water_stress = np.clip(soil_water / 200, 0, 1)
            ET = Kc * ET0 * water_stress
            
            field_capacity = 250
            runoff = max(0, rainfall - (field_capacity - soil_water))
            drainage = 0.05 * max(0, soil_water - field_capacity)
            
            soil_water = np.clip(soil_water + rainfall + irrigation - ET - runoff - drainage, 0, 400)
            
            # Store data
            data.append({
                'date': date,
                'planting_date': planting_date,
                'plot_id': plot_id,
                'variety': variety,
                'soil_texture': soil_texture,
                'soil_pH': soil_pH,
                'soil_N': soil_N,
                'tmax': tmax,
                'tmin': tmin,
                'rainfall': rainfall,
                'solar_radiation': solar_radiation,
                'fertilizer_N': fertilizer_N,
                'irrigation': irrigation,
                'LAI': LAI,
                'biomass': biomass,
                'soil_water': soil_water,
                'days_since_planting': day
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} samples and saved to '{output_file}'")
    
    return df

# ============================================
# 7. MAIN EXECUTION
# ============================================
def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED PHYSICS-INFORMED NEURAL NETWORK FOR CROP MODELING")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if not os.path.exists('crop_data.csv'):
        print("\nGenerating synthetic crop data...")
        generate_synthetic_crop_data(n_samples=5000, output_file='crop_data.csv')
    
    # Load data
    print("\nLoading data...")
    dataset = CropDataset('crop_data.csv')
    print(f"Total samples: {len(dataset)}")
    print(f"Feature dimension: {dataset.X.shape[1]}")
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Initialize model
    print("\nInitializing Enhanced PINN model...")
    model = EnhancedCropPINN(
        input_dim=dataset.X.shape[1],
        hidden_dims=[256, 128, 64],
        output_dim=3
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    history = train_pinn_enhanced(
        model, train_loader, val_loader,
        epochs=100,  # lower for testing
        device=device
    )
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load('best_pinn_model_enhanced.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.6f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    predictions, actuals, metrics = evaluate_model(model, test_loader, dataset, device=device)
    
    # Uncertainty quantification
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    mean_pred, std_pred, actuals_unc = predict_with_uncertainty(
        model, test_loader, dataset, device=device, n_samples=30
    )
    
    print("\nMean Uncertainty (±2σ):")
    print(f"  LAI: ±{2*np.mean(std_pred[:, 0]):.3f}")
    print(f"  Biomass: ±{2*np.mean(std_pred[:, 1]):.1f} kg/ha")
    print(f"  Soil Water: ±{2*np.mean(std_pred[:, 2]):.1f} mm")
    
    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Plotting training history...")
    plot_training_history(history)
    
    print("2. Plotting prediction scatter plots...")
    plot_predictions(predictions, actuals, metrics)
    
    print("3. Plotting predictions with uncertainty...")
    plot_uncertainty(mean_pred, std_pred, actuals_unc, sample_indices=range(min(200, len(actuals_unc))))
    
    print("4. Plotting residual analysis...")
    plot_residuals(predictions, actuals)
    
    print("5. Plotting time series predictions...")
    plot_time_series(predictions, actuals, dataset, test_dataset, num_samples=3)
    
    print("6. Plotting learned physics parameters...")
    plot_learned_parameters(model)
    
    print("\n" + "="*80)
    print("FINAL LEARNED PHYSICS PARAMETERS")
    print("="*80)
    print(f"RUE (Radiation Use Efficiency): {model.RUE.item():.3f} g/MJ")
    print(f"k (Light Extinction Coefficient): {model.k.item():.3f}")
    print(f"Kc (Crop Coefficient): {model.Kc.item():.3f}")
    print(f"SLA (Specific Leaf Area): {model.SLA.item():.4f} m²/g")
    print(f"T_base (Base Temperature): {model.T_base.item():.1f}°C")
    print(f"T_opt (Optimal Temperature): {model.T_opt.item():.1f}°C")

if __name__ == "__main__":
    main()
