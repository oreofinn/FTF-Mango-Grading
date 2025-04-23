# FTF-Mango-Grading

FROM TREES TO TRADE: INTEGRATING MACHINE LEARNING-BASED MANGO GRADING WITH COST-BENEFIT ANALYTICS FOR SUSTAINABLE MARKET COMPETITIVENESS AND PROFIT MAXIMIZATION

# Overview

This repository contains the code and resources for a real‑time mango grading and cost‑benefit analytics system running on a Raspberry Pi. The system uses a TensorFlow/Keras convolutional neural network (CNN) to classify mangoes into Grade A, B, C, or Rejected, computes a defect percentage via OpenCV color segmentation, and then feeds the counts and weights into a lightweight pandas‑based pipeline to calculate gross revenue, production cost, net profit, and profit contribution for each grade.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Requirements
- Hardware
- Raspberry Pi 5 (or Pi 4 with sufficient USB-C power)
- Raspberry Pi Camera Module 3 (CSI interface)

Optional: Weight sensors and servo motors for sorting mechanism

### Software
- **Operating System:** Raspberry Pi OS (64-bit recommended)
- **Python:** 3.8 or higher
- **System packages:**
  ```bash
  sudo apt update
  sudo apt install -y python3-pip python3-venv libatlas-base-dev libopencv-dev

## Installation
1. Clone this repository
   ```bash
   git clone https://github.com/oreoffinn/FTF-Mango-Grading.git
   cd FTF-Mango-Grading
2. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. Install Python dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

## Configuration
1. Place your trained CNN model files in the project root, named cnn_<variety>.keras (e.g., cnn_apple_mango.keras).

2. Ensure your dataset folder (mango_dataset/) is present if you want to retrain or evaluate on local images.

## Usage
1. Start the Flask server
   ```bash
   export FLASK_APP=app.py
   flask run --host=0.0.0.0 --port=5000
   or simply:
   python app.py
2. **Access the dashboard**
   Open a browser and navigate to http://IP-address>:5000 to view real‑time grading cards, defect percentages.

## Project Structure
```bash
mango_grading/
├── camera_with_variety_ui.py            # UI for selecting and visualizing mango variety
├── naming.py                            # Utility for consistent variety naming
├── inflation_rates_1994_2025.csv        # Historical inflation data
├── regenerate_inflation_files.py        # Script to update inflation datasets
├── train_all_varieties.py               # Train CNN models for each mango variety
├── train_srp_model.py                   # Train SRP recommendation model
├── mango_model.pkl                      # Serialized baseline model (pickle)
├── cnn_apple_mango.keras                # Trained CNN for Apple mango classification
├── cnn_carabao_mango.keras              # Trained CNN for Carabao mango classification
├── cnn_indian_mango.keras               # Trained CNN for Indian mango classification
├── cnn_pico_mango.keras                 # Trained CNN for Pico mango classification
├── mango_dataset/                       # Raw image folders (managed by Git LFS)
│   ├── APPLE MANGO/
│   ├── CARABAO MANGO/
│   ├── INDIAN MANGO/
│   └── PICO MANGO/
└── README.md                            # This file
   
