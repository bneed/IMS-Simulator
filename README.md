# IMS Physics Pro

A comprehensive Ion Mobility Spectrometry (IMS) simulation and analysis platform with both Free and Pro tiers.

## Features

### 🎯 Simulation
- Multi-ion IMS spectrum simulation
- Configurable drift tube parameters
- Realistic peak modeling with diffusion broadening
- Export simulation results

### 📊 Analysis
- Upload and analyze IMS spectra
- Peak detection and characterization
- Baseline correction
- Multiple scan aggregation (mean, max, min, sum)
- Drift time window filtering

### 📚 Library Management (Pro)
- Reference compound library
- Import/export CSV data
- Compound search and filtering
- Add compounds from analysis results

### 🤖 Machine Learning (Pro)
- K0 mobility prediction
- Chemical family classification
- Train models on your data
- Uncertainty quantification

### 🔬 Simulation Laboratory (Pro)
- Unknown compound characterization
- Library matching
- ML-based predictions
- Staging area for unknowns

### 📐 Visualization (Pro)
- 3D drift tube models
- Electric field visualizations
- Temperature heatmaps
- Ion trajectory animations

### 🚀 Trajectories (Pro)
- Ion trajectory simulation
- Multi-segment tube modeling
- Animation controls
- Export trajectory data

## Installation

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/ims-physics-pro.git
cd ims-physics-pro

# Install dependencies
pip install -r requirements.txt

# Run the application
cd apps
streamlit run app_pro.py
```

### Web Deployment
The application is designed for easy web deployment:

1. **Streamlit Community Cloud** (Recommended)
   - Connect your GitHub repository
   - Automatic deployment
   - Free hosting

2. **Heroku**
   - Add `Procfile` with: `web: streamlit run apps/app_pro.py --server.port=$PORT --server.address=0.0.0.0`
   - Deploy from GitHub

3. **Railway**
   - Connect GitHub repository
   - Automatic Python detection
   - One-click deployment

## Usage

### Free Version
- Basic simulation (up to 3 ions)
- Spectrum analysis
- Peak detection
- Export results

### Pro Version
- All Free features
- Advanced simulation (up to 12 ions)
- Library management
- Machine learning
- Visualization tools
- Trajectory simulation

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- See `requirements.txt` for full dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue on GitHub.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.