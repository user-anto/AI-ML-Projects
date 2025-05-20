# CO₂ Emissions Forecasting: USA vs India

## Overview

This project implements a time series forecasting pipeline to predict CO₂ emissions for the USA and India over the next 30 years and predict out when the curves will intersect. It consists of:

* An **LSTM ensemble** for the USA dataset.
* A **polynomial regression** (degree 2) model for the India dataset.

All data is loaded from an Excel file (`USA.xlsx`) with two sheets:

* `Sheet1`: USA CO2 emissions data.
* `Sheet2`: India CO2 emissions data.

## Requirements

* Python 3.7+
* pandas
* numpy
* matplotlib
* scikit-learn
* torch (PyTorch)

## Data Loading

```python
# USA data
df1 = pd.read_excel(
    'USA.xlsx', sheet_name='Sheet1', usecols=[1,2,5],
    names=['YEAR','CO2 PRODUCTIONS IN TONS','TOTAL POPULATION'], header=1
)

# India data (same file, Sheet2)
df2 = pd.read_excel(
    'USA.xlsx', sheet_name='Sheet2', usecols=[1,2,5],
    names=['YEAR','CO2 PRODUCTIONS IN TONS','TOTAL POPULATION'], header=1
)
```

## USA LSTM Ensemble

### 1. Preprocessing & Sequence Creation

* Extract CO₂ series and normalize with `MinMaxScaler`.
* Use a sliding window of length `seq_length = 5` to create input (`X`) and target (`y`) sequences.

```python
emissions = df1['CO2 PRODUCTIONS IN TONS'].values.astype(float).reshape(-1,1)
usa_scaler = MinMaxScaler()
em_norm = usa_scaler.fit_transform(emissions)
X, y = create_sequences(em_norm, seq_length=5)
X_train_usa = torch.from_numpy(X).float()
y_train_usa = torch.from_numpy(y).float()
```

Function to create sequences:

```python
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)
```

### 2. Model Architecture

* LSTM with:

  * `input_size=1`
  * `hidden_size=16`
  * `num_layers=1`
* Fully connected layer mapping last hidden state to a single output.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
```

### 3. Training Procedure

* Ensemble of `n_models = 5`, each initialized with `torch.manual_seed(2)`.
* Loss: Mean Squared Error (`nn.MSELoss`).
* Optimizer: `AdamW` with `lr=0.001` and `weight_decay=1e-7`.
* Early stopping when training loss < `0.0005`.

```python
def train_ensemble(models, X_train, y_train,
                   epochs=1500, lr=0.001,
                   decay=1e-7, early_stopping=0.0005):
    criterion = nn.MSELoss()
    for i, model in enumerate(models):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        for epoch in range(epochs):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            if loss.item() < early_stopping:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

Train call:

```python
torch.manual_seed(2)
usa_models = [LSTMModel() for _ in range(5)]
train_ensemble(usa_models, X_train_usa, y_train_usa,
               epochs=sys.maxsize, early_stopping=0.0005)
```

### 4. Forecasting

* Predict next `prediction_years = 30` years iteratively:

  * Use last `seq_length` normalized values as input to each model.
  * Append each prediction to the sequence.
  * Inverse-transform to original scale.
* Compute mean across the ensemble.

```python
last_seq = em_norm[-seq_length:].tolist()
usa_future_emissions_all = [...]  # predictions from each model
usa_future_emissions_mean = np.mean(usa_future_emissions_all, axis=0)
```

## India Polynomial Regression

### 1. Train/Test Split

* Features: `YEAR`
* Target: `CO2 PRODUCTIONS IN TONS`
* Split: 80% train, 20% test, no shuffle.

```python
X = df2['YEAR'].values.reshape(-1,1)
y = df2['CO2 PRODUCTIONS IN TONS'].values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
```

### 2. Pipeline & Evaluation

* Polynomial features of degree `2` (no bias).
* Standard scaling of features.
* Linear regression.

```python
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])
poly_model.fit(X_train, y_train)
test_score = poly_model.score(X_test, y_test)
print(f"Polynomial regressor R² on test: {test_score:.4f}")
```

### 3. Forecasting

* Predict next `prediction_years = 30` years:

```python
last_year = int(df2['YEAR'].iloc[-1])
future_years = np.arange(last_year+1, last_year+1+prediction_years).reshape(-1,1)
india_future_emissions = poly_model.predict(future_years)
```

## Plotting Results

```python
plt.figure(figsize=(10,6))
plt.plot(total_years, final_usa_data, '-', label='USA Forecast')
plt.plot(total_years, final_india_data, '-', label='India Forecast')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (Tons)')
plt.title('CO₂ Emissions Prediction: USA vs India')
plt.legend()
plt.savefig('co2_emissions_forecast.png', dpi=300)
plt.show()
```

* `total_years`: concatenation of historical and future years.
* `final_usa_data` & `final_india_data`: concatenation of historical and forecasted values.

* View `co2_emissions_forecast.png` for the combined forecast plot.

---