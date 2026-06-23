import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import warnings
from sklearn.model_selection import GroupShuffleSplit

def group_split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    groups = dataset.data['plot_id'].values
    indices = np.arange(len(dataset))

    gss_train = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        random_state=seed
    )

    train_idx, temp_idx = next(gss_train.split(indices, groups=groups))

    temp_groups = groups[temp_idx]

    val_fraction = val_ratio / (1.0 - train_ratio)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction,
        random_state=seed
    )

    val_relative_idx, test_relative_idx = next(
        gss_val.split(temp_idx, groups=temp_groups)
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

warnings.filterwarnings('ignore')

# ============================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================
class CropDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Convert dates to days since planting
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['planting_date'] = pd.to_datetime(self.data['planting_date'])
        self.data['days_since_planting'] = (
            self.data['date'] - self.data['planting_date']
        ).dt.days

        # Sort data by plot and time before calculating cumulative features
        self.data = self.data.sort_values(['plot_id', 'days_since_planting']).reset_index(drop=True)

# Average temperature
        self.data['temp_avg'] = (self.data['tmax'] + self.data['tmin']) / 2

# Growing Degree Days, base temperature = 10°C
        base_temp = 10.0
        self.data['daily_gdd'] = np.maximum(self.data['temp_avg'] - base_temp, 0)

# Cumulative features calculated separately for each plot
        self.data['growing_degree_days'] = self.data.groupby('plot_id')['daily_gdd'].cumsum()
        self.data['cumulative_rainfall'] = self.data.groupby('plot_id')['rainfall'].cumsum()
        self.data['cumulative_irrigation'] = self.data.groupby('plot_id')['irrigation'].cumsum()
        self.data['cumulative_fertilizer_N'] = self.data.groupby('plot_id')['fertilizer_N'].cumsum()
                
        # Encode categorical variables
        self.data['soil_texture_encoded'] = pd.Categorical(self.data['soil_texture']).codes
        self.data['variety_encoded'] = pd.Categorical(self.data['variety']).codes
        
        # Define feature columns
        self.weather_features = ['tmax', 'tmin', 'rainfall', 'solar_radiation']
        self.soil_features = ['soil_pH', 'soil_N', 'soil_texture_encoded']
        self.management_features = ['fertilizer_N', 'irrigation', 'variety_encoded']
        self.time_features = ['days_since_planting','growing_degree_days','cumulative_rainfall','cumulative_irrigation','cumulative_fertilizer_N']
        self.targets = ['LAI', 'biomass', 'soil_water', 'yield']
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        all_features = (self.weather_features + self.soil_features + 
                       self.management_features + self.time_features)
        
        self.X = self.scaler_X.fit_transform(self.data[all_features])
        
        self.scaler_y.fit(self.data[self.targets])
        self.y = self.data[self.targets].values  # Original physical units
        
        self.weather_raw = self.data[self.weather_features].values
        self.irrigation_raw = self.data['irrigation'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        curr_row = self.data.iloc[idx]
        plot_id = curr_row['plot_id']
        day = curr_row['days_since_planting']
        
        if idx + 1 < len(self.data) and self.data.iloc[idx + 1]['plot_id'] == plot_id:
            next_idx = idx + 1
            has_next = 1.0
        else:
            next_idx = idx
            has_next = 0.0
            
        return {
            'X': torch.FloatTensor(self.X[idx]),
            'y': torch.FloatTensor(self.y[idx]),
            'weather': torch.FloatTensor(self.weather_raw[idx]),
            'irrigation': torch.FloatTensor([self.irrigation_raw[idx]]),
            'time': torch.FloatTensor([day]),
            'X_next': torch.FloatTensor(self.X[next_idx]),
            'weather_next': torch.FloatTensor(self.weather_raw[next_idx]),
            'irrigation_next': torch.FloatTensor([self.irrigation_raw[next_idx]]),
            'has_next': torch.FloatTensor([has_next])
        }

# ============================================
# 2. ENHANCED PINN MODEL ARCHITECTURE
# ============================================
class EnhancedCropPINN(nn.Module):
    def __init__(self, input_dim=11, hidden_dims=[256, 128, 64], output_dim=4):
        super(EnhancedCropPINN, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.LayerNorm(hidden_dims[0])
        
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.LayerNorm(hidden_dims[1])
        
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.LayerNorm(hidden_dims[2])
        
        self.skip = nn.Linear(input_dim, hidden_dims[2])
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
        self.RUE_raw = nn.Parameter(torch.tensor(0.0))    # Maps to 1.5 - 4.5, init center 3.0
        self.k_raw = nn.Parameter(torch.tensor(0.0))      # Maps to 0.4 - 0.8, init center 0.6
        self.Kc_raw = nn.Parameter(torch.tensor(0.0))     # Maps to 0.9 - 1.3, init center 1.1
        self.SLA_raw = nn.Parameter(torch.tensor(0.0))    # Maps to 0.018 - 0.030, init center 0.024
        self.T_base_raw = nn.Parameter(torch.tensor(0.0)) # Maps to 8 - 12, init center 10.0
        self.T_opt_raw = nn.Parameter(torch.tensor(0.0))  # Maps to 23 - 28, init center 25.5
        self.HI_raw = nn.Parameter(torch.tensor(0.0))  # Harvest index, maps to 0.35 - 0.60

        
    @property
    def RUE(self): return 1.5 + (4.5 - 1.5) * torch.sigmoid(self.RUE_raw)
    @property
    def k(self): return 0.4 + (0.8 - 0.4) * torch.sigmoid(self.k_raw)
    @property
    def Kc(self): return 0.9 + (1.3 - 0.9) * torch.sigmoid(self.Kc_raw)
    @property
    def SLA(self): return 0.018 + (0.030 - 0.018) * torch.sigmoid(self.SLA_raw)
    @property
    def T_base(self): return 8.0 + (12.0 - 8.0) * torch.sigmoid(self.T_base_raw)
    @property
    def T_opt(self): return 23.0 + (28.0 - 23.0) * torch.sigmoid(self.T_opt_raw)
    @property
    def HI(self): return 0.35 + (0.60 - 0.35) * torch.sigmoid(self.HI_raw)
        
    def forward(self, x):
        
        h1 = self.dropout(self.activation(self.bn1(self.layer1(x))))
        h2 = self.dropout(self.activation(self.bn2(self.layer2(h1))))
        h3 = self.activation(self.bn3(self.layer3(h2)))
        
        h3 = h3 + self.skip(x)
        output = self.output_layer(h3)
        
        LAI = 8.0 * torch.sigmoid(output[:, 0])           # Max LAI = 8
        biomass = 20000.0 * torch.sigmoid(output[:, 1])   # Max biomass = 20000 kg/ha
        soil_water = 400.0 * torch.sigmoid(output[:, 2])  # Max soil water = 400 mm
        yield_pred = 12000.0 * torch.sigmoid(output[:, 3])  # Max grain yield = 12000 kg/ha


        """yield_pred = 8000.0 * torch.sigmoid(output[:, 3])     # for moderate yield crops
yield_pred = 12000.0 * torch.sigmoid(output[:, 3])    # for high yielding cereal crops
yield_pred = 15000.0 * torch.sigmoid(output[:, 3])    # for very high biomass/yield systems"""

        return torch.stack([LAI, biomass, soil_water, yield_pred], dim=1)
    
    def temperature_stress(self, temp_avg):
        stress = torch.zeros_like(temp_avg)
        mask2 = (temp_avg >= self.T_base) & (temp_avg <= self.T_opt)
        stress[mask2] = (temp_avg[mask2] - self.T_base) / (self.T_opt - self.T_base)
        mask3 = (temp_avg > self.T_opt) & (temp_avg <= 40.0)
        stress[mask3] = (40.0 - temp_avg[mask3]) / (40.0 - self.T_opt)
        return stress
    
    def enhanced_physics_loss(self, predictions, predictions_next, weather, irrigation, time_steps, has_next):
        LAI = predictions[:, 0]
        biomass = predictions[:, 1]
        soil_water = predictions[:, 2]
        yield_pred = predictions[:, 3]

        LAI_next = predictions_next[:, 0]
        biomass_next = predictions_next[:, 1]
        soil_water_next = predictions_next[:, 2]
        yield_next = predictions_next[:, 3]
        
        mask = (has_next.squeeze() > 0.5)
        solar_radiation, rainfall = weather[:, 3], weather[:, 2]
        temp_avg = (weather[:, 0] + weather[:, 1]) / 2
        temp_stress = self.temperature_stress(temp_avg)
        
        # 1. BIOMASS GROWTH
        PAR = solar_radiation * 0.5
        IPAR = PAR * (1 - torch.exp(-self.k * LAI))
        growth_potential = self.RUE * IPAR * temp_stress
        
        physics_loss_growth = torch.tensor(0.0, device=predictions.device)
        active_growth_mask = mask & (biomass < 19500)
        if torch.any(active_growth_mask):
            biomass_diff = biomass_next[active_growth_mask] - biomass[active_growth_mask]
            growth_potential_scaled = growth_potential[active_growth_mask] * 10.0
            physics_loss_growth = torch.mean(((biomass_diff - growth_potential_scaled) / 100.0)**2)
        
        # 2. LAI-BIOMASS RELATIONSHIP
        canopy_mask = (LAI < 7.5) & (time_steps.squeeze() < 60)
        physics_loss_lai_biomass = torch.tensor(0.0, device=predictions.device)
        if torch.any(canopy_mask):
            time_normalized = torch.clamp(time_steps.squeeze()[canopy_mask] / 100.0, 0, 1)
            partition_coef = 0.6 - 0.3 * time_normalized
            biomass_g_m2 = biomass[canopy_mask] * 0.1
            LAI_from_biomass = biomass_g_m2 * self.SLA * partition_coef
            physics_loss_lai_biomass = torch.mean(((LAI[canopy_mask] - LAI_from_biomass) / 4.0)**2)
        
        # 3. WATER BALANCE
        temp_range = torch.clamp(weather[:, 0] - weather[:, 1], min=0.1)
        ET0 = 0.0023 * (temp_avg + 17.8) * torch.sqrt(temp_range) * solar_radiation * 0.408
        water_stress = torch.clamp(soil_water / 200.0, 0.0, 1.0)
        ET_actual = self.Kc * ET0 * water_stress
        
        field_capacity = 250.0
        storage_available = torch.clamp(field_capacity - soil_water, min=0.0)
        runoff = torch.relu(rainfall - storage_available)
        drainage = 0.05 * torch.relu(soil_water - field_capacity)
        
        water_balance_theory = rainfall + irrigation.squeeze() - ET_actual - runoff - drainage
        
        physics_loss_water = torch.tensor(0.0, device=predictions.device)
        if torch.any(mask):
            water_diff = soil_water_next[mask] - soil_water[mask]
            physics_loss_water = torch.mean(((water_diff - water_balance_theory[mask]) / 5.0)**2)
        
        # 4. MONOTONICITY
        physics_loss_monotonic = torch.tensor(0.0, device=predictions.device)
        if torch.any(mask):
            biomass_decrease = torch.relu(biomass[mask] - biomass_next[mask])
            physics_loss_monotonic = torch.mean((biomass_decrease / 100.0)**2)
        
        # 5. SENESCENCE
        physics_loss_senescence = torch.tensor(0.0, device=predictions.device)
        late_mask = (time_steps.squeeze() > 90) & mask
        if torch.any(late_mask):
            LAI_increase = torch.relu(LAI_next[late_mask] - LAI[late_mask])
            physics_loss_senescence = torch.mean(LAI_increase**2)

        # 6. YIELD FORMATION PHYSICS
        # 6. YIELD FORMATION PHYSICS
physics_loss_yield = torch.tensor(0.0, device=predictions.device)

time_flat = time_steps.squeeze()

# Grain filling usually starts after vegetative growth
grain_filling_mask = (time_flat > 60) & mask

if torch.any(grain_filling_mask):

    theoretical_yield = self.HI * biomass[grain_filling_mask]

    # Yield should approach harvest-index-based biomass fraction
    physics_loss_yield_relation = torch.mean(
        ((yield_pred[grain_filling_mask] - theoretical_yield) / 1000.0) ** 2
    )

    # Yield should not decrease during grain filling
    yield_decrease = torch.relu(
        yield_pred[grain_filling_mask] - yield_next[grain_filling_mask]
    )

    physics_loss_yield_monotonic = torch.mean((yield_decrease / 100.0) ** 2)

    physics_loss_yield = (
        physics_loss_yield_relation +
        0.5 * physics_loss_yield_monotonic
    )

# Early crop stage yield should be close to zero
early_yield_mask = time_flat < 40

physics_loss_early_yield = torch.tensor(0.0, device=predictions.device)

if torch.any(early_yield_mask):

    physics_loss_early_yield = torch.mean((yield_pred[early_yield_mask] / 200.0) ** 2)
        
    total_physics_loss = (
        1.0 * physics_loss_growth +
        0.5 * physics_loss_water +
        0.3 * physics_loss_lai_biomass +
        0.2 * physics_loss_monotonic +
        0.1 * physics_loss_senescence +
        0.5 * physics_loss_yield +
        0.2 * physics_loss_early_yield
    )
        
        components = {
            'growth': physics_loss_growth.item(),
            'water': physics_loss_water.item(),
            'lai_biomass': physics_loss_lai_biomass.item(),
            'monotonic': physics_loss_monotonic.item(),
            'senescence': physics_loss_senescence.item(),
            'yield': physics_loss_yield.item(),
            'early_yield': physics_loss_early_yield.item(),
            'energy': 0.0,
        }
        
        return total_physics_loss, components
    
    def boundary_loss(self, predictions, time_steps):

    loss = torch.tensor(0.0, device=predictions.device)

    time_steps = time_steps.squeeze()

    initial_mask = time_steps < 5

    if torch.any(initial_mask):

        # LAI should be near zero initially
        loss += torch.mean((predictions[initial_mask, 0] / 1.0) ** 2)

        # Biomass should be near 50 kg/ha initially,
        # but use softer scaling to avoid suppressing biomass learning
        loss += torch.mean(((predictions[initial_mask, 1] - 50.0) / 500.0) ** 2)

        # Yield should be near zero initially
        loss += torch.mean((predictions[initial_mask, 3] / 200.0) ** 2)

    return loss

# ============================================
# 3. ENHANCED TRAINING FUNCTION
# ============================================
def train_pinn_enhanced(model, train_loader, val_loader, scaler_y, epochs=300, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'physics_loss': [], 'boundary_loss': [],
        'learning_rate': [], 'components': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 100
    y_std = torch.FloatTensor(scaler_y.scale_).to(device)
    
    print(f"Training on device: {device}")
    print("=" * 80)
    
    for epoch in range(epochs):

    # Physics warm-up
        warmup_epochs = 30

        if epoch < warmup_epochs:
            lambda_physics = 0.0
        else:
            lambda_physics = min(0.1,((epoch - warmup_epochs) / max(1, epochs - warmup_epochs)) * 0.1
        )

    # Reduce boundary effect
    lambda_boundary = 0.005

    # Dynamic yield weight
    yield_weight = min(0.5, 0.1 + epoch / epochs * 0.4)

    # IMPORTANT:
    # Biomass weight increased from 0.1 to 1.5
    weights = torch.tensor([1.0, 1.5, 0.5, yield_weight], device=device)
        
        model.train()
        train_loss_total, data_loss_total, physics_loss_total, boundary_loss_total = 0, 0, 0, 0
        components_epoch = {
            'growth': 0,
            'water': 0,
            'lai_biomass': 0,
            'monotonic': 0,
            'senescence': 0,
            'yield': 0,
            'early_yield': 0,
            'energy': 0
        }
        
        for batch in train_loader:
            X, y_true = batch['X'].to(device), batch['y'].to(device)
            weather, irrigation = batch['weather'].to(device), batch['irrigation'].to(device)
            time, has_next = batch['time'].to(device), batch['has_next'].to(device)
            X_next = batch['X_next'].to(device)
            
            optimizer.zero_grad()
            y_pred, y_pred_next = model(X), model(X_next)
            

            huber = torch.nn.SmoothL1Loss(reduction='none')

            loss_each = huber(y_pred / y_std, y_true / y_std)

            data_loss = torch.mean(weights * torch.mean(loss_each, dim=0))
            physics_loss, components = model.enhanced_physics_loss(y_pred, y_pred_next, weather, irrigation, time, has_next)
            boundary_loss = model.boundary_loss(y_pred, time)
            
            total_loss = data_loss + lambda_physics * physics_loss + lambda_boundary * boundary_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += total_loss.item()
            data_loss_total += data_loss.item()
            physics_loss_total += physics_loss.item()
            boundary_loss_total += boundary_loss.item()
            for k in components: components_epoch[k] += components[k]
        
        for k in components_epoch: components_epoch[k] /= len(train_loader)
        
        model.eval()
        val_loss_total = 0
        val_data_loss_total = 0

        
        with torch.no_grad():
            for batch in val_loader:
                X, y_true = batch['X'].to(device), batch['y'].to(device)
                weather, irrigation = batch['weather'].to(device), batch['irrigation'].to(device)
                time, has_next = batch['time'].to(device), batch['has_next'].to(device)
                X_next = batch['X_next'].to(device)
                
                y_pred, y_pred_next = model(X), model(X_next)
                
                weights = torch.tensor([1.0, 1.0, 0.5, yield_weight], device=device)
                normalized_error = (y_pred - y_true) / y_std

                huber = torch.nn.SmoothL1Loss(reduction='none')
                loss_each = huber(y_pred / y_std, y_true / y_std)
                data_loss = torch.mean(weights * torch.mean(loss_each, dim=0))
                
                physics_loss, _ = model.enhanced_physics_loss(y_pred, y_pred_next, weather, irrigation, time, has_next)
                boundary_loss = model.boundary_loss(y_pred, time)
                total_loss = data_loss + lambda_physics * physics_loss + lambda_boundary * boundary_loss
                val_loss_total += total_loss.item()
                
                val_data_loss_total += data_loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_data_loss = val_data_loss_total / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['data_loss'].append(data_loss_total / len(train_loader))
        history['physics_loss'].append(physics_loss_total / len(train_loader))
        history['boundary_loss'].append(boundary_loss_total / len(train_loader))
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['components'].append(components_epoch.copy())
        
        scheduler.step()
        
        if avg_val_data_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss, 'history': history,
                'val_loss': avg_val_loss,
                'val_data_loss': avg_val_data_loss,
            }, 'best_pinn_model_enhanced.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            break
    
    return history

# ============================================
# 4. EVALUATION & VISUALIZATION FUNCTIONS
# ============================================
def evaluate_model(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            predictions.append(model(batch['X'].to(device)).cpu().numpy())
            actuals.append(batch['y'].numpy())
            
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    mae = np.mean(np.abs(predictions - actuals), axis=0)
    rmse = np.sqrt(np.mean((predictions - actuals)**2, axis=0))
    
    r2 = []
    for i in range(4):
        ss_res = np.sum((actuals[:, i] - predictions[:, i])**2)
        ss_tot = np.sum((actuals[:, i] - np.mean(actuals[:, i]))**2)
        r2.append(1 - (ss_res / ss_tot))
    
    print("\n" + "="*100)
    print("MODEL PERFORMANCE METRICS")
    print("="*100)
    print(f"{'Metric':<15} {'LAI':<15} {'Biomass':<20} {'Soil Water':<20} {'Yield':<20}")
    print("-"*100)
    print(f"{'MAE':<15} {mae[0]:<15.3f} {mae[1]:<20.1f} {mae[2]:<20.1f} {mae[3]:<20.1f}")
    print(f"{'RMSE':<15} {rmse[0]:<15.3f} {rmse[1]:<20.1f} {rmse[2]:<20.1f} {rmse[3]:<20.1f}")
    print(f"{'R²':<15} {r2[0]:<15.3f} {r2[1]:<20.3f} {r2[2]:<20.3f} {r2[3]:<20.3f}")
    print("="*100)
    return predictions, actuals

def predict_with_uncertainty(model, test_loader, device='cpu', n_samples=30):
    model = model.to(device)
    model.train()
    predictions_samples, actuals = [], []
    for sample in range(n_samples):
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                if sample == 0: actuals.append(batch['y'].numpy())
                preds.append(model(batch['X'].to(device)).cpu().numpy())
        predictions_samples.append(np.vstack(preds))
    return np.mean(predictions_samples, axis=0), np.std(predictions_samples, axis=0), np.vstack(actuals)

def plot_training_history(history):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['data_loss'], label='Data Loss', linewidth=2)
    ax2.plot(history['physics_loss'], label='Physics Loss', linewidth=2)
    ax2.plot(history['boundary_loss'], label='Boundary Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['learning_rate'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot([comp['growth'] for comp in history['components']], linewidth=2, color='brown')
    ax4.set_xlabel('Epoch', fontsize=12); ax4.set_ylabel('Loss', fontsize=12); ax4.set_title('Growth Physics Loss', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot([comp['water'] for comp in history['components']], linewidth=2, color='blue')
    ax5.set_xlabel('Epoch', fontsize=12); ax5.set_ylabel('Loss', fontsize=12); ax5.set_title('Water Balance Physics Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot([comp['lai_biomass'] for comp in history['components']], linewidth=2, color='green')
    ax6.set_xlabel('Epoch', fontsize=12); ax6.set_ylabel('Loss', fontsize=12); ax6.set_title('LAI-Biomass Physics Loss', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot([comp['monotonic'] for comp in history['components']], linewidth=2, color='purple')
    ax7.set_xlabel('Epoch', fontsize=12); ax7.set_ylabel('Loss', fontsize=12); ax7.set_title('Monotonicity Physics Loss', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot([comp['senescence'] for comp in history['components']], linewidth=2, color='orange')
    ax8.set_xlabel('Epoch', fontsize=12); ax8.set_ylabel('Loss', fontsize=12); ax8.set_title('Senescence Physics Loss', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Enhanced PINN', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(predictions, actuals):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)', 'Yield (kg/ha)']
    colors = ['green', 'brown', 'blue', 'goldenrod']
    for i, (var, color) in enumerate(zip(variables, colors)):
        axes[i].scatter(actuals[:, i], predictions[:, i], alpha=0.5, color=color, edgecolors='black', linewidth=0.5)
        min_v, max_v = actuals[:, i].min(), actuals[:, i].max()
        axes[i].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Prediction')
        axes[i].set_xlabel(f'Actual {var}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(f'Predicted {var}', fontsize=12, fontweight='bold')
        axes[i].set_title(f'{var} Predictions', fontsize=14, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_scatter.png', dpi=300)
    plt.close()

def plot_uncertainty(mean_pred, std_pred, actuals):

    sample_indices = range(min(200, len(actuals)))

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)', 'Yield (kg/ha)']
    colors = ['green', 'brown', 'blue', 'goldenrod']

    for i, (var, color) in enumerate(zip(variables, colors)):

        x = np.array(list(sample_indices))

        axes[i].plot(
            x,
            actuals[x, i],
            'o',
            color='black',
            label='Actual',
            markersize=4,
            alpha=0.6
        )

        axes[i].plot(
            x,
            mean_pred[x, i],
            '-',
            color=color,
            linewidth=2,
            label='Predicted Mean'
        )

        axes[i].fill_between(
            x,
            mean_pred[x, i] - 2 * std_pred[x, i],
            mean_pred[x, i] + 2 * std_pred[x, i],
            alpha=0.3,
            color=color,
            label='95% Confidence Interval'
        )

        axes[i].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(var, fontsize=12, fontweight='bold')
        axes[i].set_title(f'{var} Predictions with Uncertainty', fontsize=14, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('predictions_uncertainty.png', dpi=300)
    plt.close()

def plot_residuals(predictions, actuals):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)', 'Yield (kg/ha)']
    colors = ['green', 'brown', 'blue', 'goldenrod']
    for i, (var, color) in enumerate(zip(variables, colors)):
        residuals = actuals[:, i] - predictions[:, i]
        axes[0, i].scatter(predictions[:, i], residuals, alpha=0.5, color=color, edgecolors='black', linewidth=0.5)
        axes[0, i].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, i].set_xlabel(f'Predicted {var}', fontsize=11, fontweight='bold')
        axes[0, i].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, i].set_title(f'{var} Residuals', fontsize=12, fontweight='bold')
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
        axes[1, i].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, i].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, i].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, i].set_title(f'{var} Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residuals_analysis.png', dpi=300)
    plt.close()

def plot_time_series(predictions, actuals, dataset, test_dataset, num_samples=3):
    test_indices = test_dataset.indices
    time_data = dataset.data['days_since_planting'].values[test_indices]
    plot_ids = dataset.data['plot_id'].values[test_indices]
    unique_ids = np.unique(plot_ids)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 4*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    variables = ['LAI', 'Biomass (kg/ha)', 'Soil Water (mm)', 'Yield (kg/ha)']
    colors = ['green', 'brown', 'blue', 'goldenrod']
    
    selected_ids = np.random.choice(unique_ids, min(num_samples, len(unique_ids)), replace=False)
    for row, plot_id in enumerate(selected_ids):
        local_indices = np.where(plot_ids == plot_id)[0]
        if len(local_indices) == 0: continue
        time_subset = time_data[local_indices]
        sort_idx = np.argsort(time_subset)
        local_indices_sorted = local_indices[sort_idx]
        time_subset_sorted = time_subset[sort_idx]
        for col, (var, color) in enumerate(zip(variables, colors)):
            axes[row, col].plot(time_subset_sorted, actuals[local_indices_sorted, col], 'o-', color='black', label='Actual', markersize=5, linewidth=2)
            axes[row, col].plot(time_subset_sorted, predictions[local_indices_sorted, col], 's-', color=color, label='Predicted', markersize=5, linewidth=2, alpha=0.7)
            axes[row, col].set_xlabel('Days Since Planting', fontsize=11, fontweight='bold')
            axes[row, col].set_ylabel(var, fontsize=11, fontweight='bold')
            axes[row, col].set_title(f'{var} - Plot {plot_id}', fontsize=12, fontweight='bold')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series_predictions.png', dpi=300)
    plt.close()

def plot_learned_parameters(model):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    params = {
        'RUE': model.RUE.item(),
        'k': model.k.item(),
        'Kc': model.Kc.item(),
        'SLA': model.SLA.item(),
        'T_base': model.T_base.item(),
        'T_opt': model.T_opt.item(),
        'HI': model.HI.item(),
    }
    expected_ranges = {
        'RUE': (1.5, 4.5, 'g/MJ'),
        'k': (0.4, 0.8, ''),
        'Kc': (0.9, 1.3, ''),
        'SLA': (0.018, 0.030, 'm²/g'),
        'T_base': (8, 12, '°C'),
        'T_opt': (23, 28, '°C'),
        'HI': (0.35, 0.60, ''),
    }
#############
    for ax in axes.flat[len(params):]:
        ax.axis('off')
#############
    
    for idx, (param_name, ax) in enumerate(zip(params.keys(), axes.flat)):
        value = params[param_name]
        exp_min, exp_max, unit = expected_ranges[param_name]
        ax.barh(0, value, height=0.5, color='green', alpha=0.7, label='Learned Value')
        ax.axvline(x=exp_min, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Expected Range')
        ax.axvline(x=exp_max, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvspan(exp_min, exp_max, alpha=0.2, color='yellow')
        ax.set_xlim(exp_min - 0.2*(exp_max-exp_min), exp_max + 0.2*(exp_max-exp_min))
        ax.set_yticks([])
        ax.set_xlabel(f'{param_name} ({unit})' if unit else param_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{param_name} = {value:.3f} {unit}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle('Learned Physics Parameters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learned_parameters.png', dpi=300)
    plt.close()

# ============================================
# 5. SYNTHETIC DATA GENERATION (FOR TESTING)
# ============================================
def generate_synthetic_crop_data(n_samples=5000, output_file='crop_data.csv'):
    np.random.seed(42)
    n_plots = 50
    days_per_plot = n_samples // n_plots
    data = []
    for plot_id in range(n_plots):
        planting_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        variety = np.random.choice(['Variety_A', 'Variety_B', 'Variety_C'])
        soil_texture = np.random.choice(['Clay', 'Loam', 'Sandy'])
        soil_pH, soil_N = np.random.uniform(5.5, 7.5), np.random.uniform(50, 150)
        LAI, biomass, soil_water, grain_yield = 0.1, 50.0, 200.0, 0.0
        # Variety-specific physiological parameters


        if variety == 'Variety_A':
            RUE = 3.2
            HI = 0.50
        elif variety == 'Variety_B':
            RUE = 2.8
            HI = 0.45
        else:
            RUE = 3.0
            HI = 0.48

        # Common physiological parameters
        k = 0.6
        Kc = 1.1
        SLA = 0.024


        
        for day in range(days_per_plot):
            date = planting_date + pd.Timedelta(days=day)
            doy = date.dayofyear


            
            tmax = 25 + 5 * np.sin(2*np.pi*doy/365) + np.random.normal(0, 2)
            tmin = tmax - 10 + np.random.normal(0, 1)
            solar_radiation = 18 + 4 * np.sin(2*np.pi*doy/365) + np.random.normal(0, 1)
            rainfall = max(0, np.random.gamma(2, 2))
            fertilizer_N = 100 if day == 30 else 0
            irrigation = 20 if day % 7 == 0 and rainfall < 5 else 0
            temp_avg = (tmax + tmin) / 2
            temp_stress = max(0.0, min(1.0, (temp_avg - 10) / 15)) if temp_avg < 25 else max(0.0, (40 - temp_avg) / 15)
            PAR = solar_radiation * 0.5
            IPAR = PAR * (1 - np.exp(-k * LAI))
            daily_growth = RUE * IPAR * temp_stress * 10
            partition = 0.6 - 0.3 * (day / days_per_plot)
            LAI_growth = daily_growth * 0.1 * SLA * partition
            if day > 90: LAI_growth -= 0.05 * LAI
            LAI = max(0, min(8, LAI + LAI_growth))
            biomass = max(50, min(20000, biomass + daily_growth))

            # Grain yield formation begins after flowering/grain filling stage
            if day > 60:
                grain_partition = min(0.8, (day - 60) / 50.0)
                grain_growth = daily_growth * HI * grain_partition
                grain_yield = min(HI * biomass, grain_yield + grain_growth)
            else:
                grain_yield = 0.0

            
            ET0 = 0.0023 * (temp_avg + 17.8) * np.sqrt(max(0.1, tmax - tmin)) * solar_radiation * 0.408
            ET = Kc * ET0 * np.clip(soil_water / 200, 0, 1)
            runoff = max(0, rainfall - (250 - soil_water))
            drainage = 0.05 * max(0, soil_water - 250)
            soil_water = np.clip(soil_water + rainfall + irrigation - ET - runoff - drainage, 0, 400)
            data.append({
                'date': date, 'planting_date': planting_date, 'plot_id': plot_id,
                'variety': variety, 'soil_texture': soil_texture, 'soil_pH': soil_pH,
                'soil_N': soil_N, 'tmax': tmax, 'tmin': tmin, 'rainfall': rainfall,
                'solar_radiation': solar_radiation, 'fertilizer_N': fertilizer_N,
                'irrigation': irrigation, 'LAI': LAI, 'biomass': biomass,
                'soil_water': soil_water, 'yield': grain_yield, 'days_since_planting': day
                
            })
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return df

# ============================================
# 6. MAIN EXECUTION
# ============================================
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    regenerate_data = False

    if not os.path.exists('crop_data.csv'):
        regenerate_data = True
    else:
        existing_columns = pd.read_csv('crop_data.csv', nrows=1).columns
        if 'yield' not in existing_columns:
            print("Existing crop_data.csv does not contain 'yield' column.")
            print("Regenerating synthetic dataset with yield...")
            regenerate_data = True

    if regenerate_data:
        generate_synthetic_crop_data(5000, 'crop_data.csv')
    
    dataset = CropDataset('crop_data.csv')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = group_split_dataset(dataset)
    
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = EnhancedCropPINN(input_dim=dataset.X.shape[1])
    
    print("\nStarting Training...")
    history = train_pinn_enhanced(model, train_loader, val_loader, dataset.scaler_y, epochs=150, device=device)
    
    checkpoint = torch.load('best_pinn_model_enhanced.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on Test Set...")
    predictions, actuals = evaluate_model(model, test_loader, device=device)
    mean_pred, std_pred, actuals_unc = predict_with_uncertainty(model, test_loader, device=device, n_samples=30)
    
    print("\nGenerating all 6 Diagnostic Visualizations...")
    plot_training_history(history)
    plot_predictions(predictions, actuals)
    plot_uncertainty(mean_pred, std_pred, actuals_unc)
    plot_residuals(predictions, actuals)
    plot_time_series(predictions, actuals, dataset, test_dataset, num_samples=3)
    plot_learned_parameters(model)
    print("\nVisualizations saved successfully!")

if __name__ == "__main__":
    main()
