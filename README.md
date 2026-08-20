# 🌾 AgroPINN: Physics-Informed Crop Growth and Yield Prediction

> A Physics-Informed Neural Network (PINN) framework for predicting crop growth dynamics and crop yield by combining deep learning with agricultural domain knowledge.

## 📌 Overview

**AgroPINN** is a Physics-Informed Neural Network designed for crop growth and yield prediction. Unlike conventional machine learning models that rely only on historical data, AgroPINN integrates agricultural and crop-growth principles directly into the neural network training process.

The model uses weather, soil, crop management, and temporal information to simultaneously predict four important crop variables:

* 🌱 **Leaf Area Index (LAI)**
* 🌿 **Biomass**
* 💧 **Soil Water Content**
* 🌾 **Crop Yield**

The framework combines data-driven learning with physics-based constraints such as biomass accumulation, soil water balance, LAI–biomass relationships, crop senescence, yield formation, and boundary conditions.

---

## 🎯 Problem Statement

Crop growth and yield depend on complex interactions between environmental conditions, soil properties, crop management practices, and time.

Traditional machine learning models can learn patterns from agricultural data, but they may produce predictions that violate known biological or physical relationships.

AgroPINN addresses this limitation by incorporating crop physiology and agricultural constraints into the learning process.

The objective is to develop a model that produces predictions that are both:

* **Accurate**
* **Biologically and physically consistent**

---

## 🏗️ System Workflow

```text
Agricultural Dataset
        │
        ▼
Data Preprocessing
        │
        ├── Missing Value Handling
        ├── Categorical Encoding
        ├── Feature Engineering
        └── Z-Score Normalization
        │
        ▼
AgroPINN Neural Network
        │
        ├── 256 Neurons
        ├── 128 Neurons
        └── 64 Neurons
        │
        ▼
Physics-Informed Training
        │
        ├── Data Loss
        ├── Physics Loss
        └── Boundary Loss
        │
        ▼
Model Evaluation
        │
        ├── MAE
        ├── RMSE
        ├── R² Score
        └── Monte Carlo Dropout
        │
        ▼
Decision Support Outputs
```

---

# 📊 Dataset

The AgroPINN framework was developed using a crop growth dataset containing:

| Property         | Value                   |
| ---------------- | ----------------------- |
| Total Samples    | 5,000                   |
| Total Attributes | 18                      |
| Input Features   | 15                      |
| Target Variables | 4                       |
| Learning Task    | Multi-Output Regression |
| Application      | Precision Agriculture   |

## Input Features

### 🌦️ Weather Features

* Maximum Temperature (`Tmax`)
* Minimum Temperature (`Tmin`)
* Rainfall
* Solar Radiation

### 🌱 Soil Features

* Soil pH
* Soil Nitrogen
* Soil Texture

### 🚜 Crop Management Features

* Fertilizer Application
* Irrigation
* Crop Variety

### ⏳ Temporal Features

* Days Since Planting
* Growing Degree Days (GDD)
* Cumulative Rainfall
* Cumulative Irrigation
* Cumulative Fertilizer Application

---

# 🎯 Target Variables

The model simultaneously predicts:

| Output     | Description                                          |
| ---------- | ---------------------------------------------------- |
| LAI        | Leaf Area Index representing crop canopy development |
| Biomass    | Crop biomass accumulation                            |
| Soil Water | Soil water content                                   |
| Yield      | Final crop yield                                     |

---

# 🧠 Model Architecture

AgroPINN uses a fully connected neural network with the following architecture:

```text
Input Features (15)
        │
        ▼
Dense Layer – 256 Neurons
GELU Activation
Layer Normalization
Dropout (0.2)
        │
        ▼
Dense Layer – 128 Neurons
GELU Activation
Layer Normalization
Dropout (0.2)
        │
        ▼
Dense Layer – 64 Neurons
GELU Activation
Layer Normalization
Dropout (0.2)
        │
        ▼
Residual Skip Connection
        │
        ▼
Output Layer
        │
 ┌──────┼──────┬─────────────┐
 ▼      ▼      ▼             ▼
LAI  Biomass Soil Water    Yield
```

The model also learns physiologically meaningful crop parameters during training.

---

# ⚛️ Physics-Informed Learning

The total training objective combines data-driven learning with physics-based constraints:

```text
Total Loss =
Data Loss
+
λₚ × Physics Loss
+
λᵦ × Boundary Loss
```

## 📉 Data Loss

Huber Loss is used for supervised learning because it is more robust to noisy observations and outliers.

## 🔬 Physics Loss

The physics-informed loss incorporates agricultural relationships involving:

* Biomass accumulation
* Radiation Use Efficiency
* Solar radiation interception
* Temperature stress
* LAI–Biomass relationship
* Soil water balance
* Crop evapotranspiration
* Yield formation
* Crop senescence
* Biomass monotonicity

## 🚧 Boundary Loss

Boundary conditions are used to encourage realistic crop conditions during the initial stages of crop growth.

---

# 🌱 Learnable Physiological Parameters

AgroPINN learns several crop physiological parameters during optimization:

| Parameter | Description                  |
| --------- | ---------------------------- |
| RUE       | Radiation Use Efficiency     |
| k         | Light Extinction Coefficient |
| Kc        | Crop Coefficient             |
| SLA       | Specific Leaf Area           |
| Tbase     | Base Temperature             |
| Topt      | Optimum Temperature          |
| HI        | Harvest Index                |

The parameters are constrained within biologically realistic ranges.

---

# ⚙️ Training Strategy

The model is trained using:

* **Optimizer:** AdamW
* **Learning Rate Scheduler:** Cosine Annealing
* **Loss Function:** Huber Loss + Physics Loss + Boundary Loss
* **Regularization:** Dropout
* **Gradient Clipping**
* **Early Stopping**
* **Physics Warm-Up Strategy**
* **Group-Based Dataset Splitting**

The physics-based constraints are gradually introduced during training to allow the network to first learn the underlying data distribution.

---

# 📈 Results

The model was evaluated using:

* Mean Absolute Error (**MAE**)
* Root Mean Square Error (**RMSE**)
* Coefficient of Determination (**R²**)

## Performance Metrics

| Metric   |       LAI |   Biomass | Soil Water |     Yield |
| -------- | --------: | --------: | ---------: | --------: |
| MAE      |     0.253 |     477.6 |        4.0 |     262.2 |
| RMSE     |     0.402 |     619.5 |        5.1 |     388.4 |
| R² Score | **0.981** | **0.992** |  **0.784** | **0.981** |

The model achieved strong predictive performance for LAI, biomass, and crop yield, while soil water content showed comparatively lower—but still useful—predictive performance.

---

# 📊 Model Evaluation

The AgroPINN framework includes the following evaluation and analysis methods:

* Training and validation loss curves
* Actual vs predicted comparison
* Residual analysis
* Time-series prediction analysis
* Monte Carlo Dropout uncertainty estimation
* Learned physiological parameter analysis

Monte Carlo Dropout is used to estimate prediction uncertainty through multiple stochastic forward passes during inference.

---

# 📁 Recommended Project Structure

# 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/surajsharma23/PINN-forecasting-yield.git
cd PINN-forecasting-yield
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python demo.py
```

> Update the final run command if your repository uses a different main Python file.

---

# 🔮 Future Improvements

Possible future extensions include:

* Validation using large-scale real-world agricultural datasets
* Integration with IoT-based agricultural sensors
* Remote sensing and satellite data integration
* Support for multiple crop species
* Real-time crop monitoring
* Web-based decision-support dashboard
* Improved soil water modeling
* Deployment as a precision agriculture application

---

# 📄 Research Paper

This project is based on:

**AgroPINN: Physics-Informed Crop Growth and Yield Prediction System**

The work presents a hybrid deep learning framework that integrates Physics-Informed Neural Networks with agricultural domain knowledge for crop growth monitoring, irrigation planning, fertilizer management, and crop yield prediction.

---

# 👨‍💻 Author

**Suraj Sharma**

M.Tech – Artificial Intelligence & Data Science
Bapuji Institute of Engineering and Technology, Davangere

📧 Email: [surajsharma.officially@gmail.com](mailto:surajsharma.officially@gmail.com)

🔗 GitHub: https://github.com/surajsharma23

---

# ⭐ If you found this project useful

Consider giving the repository a **star ⭐**.
