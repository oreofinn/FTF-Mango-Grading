# FTF-Mango-Grading

FROM TREES TO TRADE: INTEGRATING MACHINE LEARNING-BASED MANGO GRADING WITH COST-BENEFIT ANALYTICS FOR SUSTAINABLE MARKET COMPETITIVENESS AND PROFIT MAXIMIZATION

# Overview

This repository contains the code and resources for a real‑time mango grading and cost‑benefit analytics system running on a Raspberry Pi. The system uses a TensorFlow/Keras convolutional neural network (CNN) to classify mangoes into Grade A, B, C, or Rejected, computes a defect percentage via OpenCV color segmentation, and then feeds the counts and weights into a lightweight pandas‑based pipeline to calculate gross revenue, production cost, net profit, and profit contribution for each grade.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Requirements
- Hardware
- Raspberry Pi 5
- Raspberry Pi Camera Module 3

### Software
- **Operating System:** Raspberry Pi OS
- **Python:** 3.8 or higher
- **Python Libraries:**
  ```bash
    Picamera2
    opencv-python           # OpenCV for image processing
    numpy                   # numerical computing
    joblib                  # model serialization
    tensorflow              # CNN inference and training
    scikit-learn            # SRP regression model
    flask                   # lightweight web framework
    flask-socketio          # real-time updates
    eventlet                # async worker for SocketIO
    pandas                  # data manipulation (reports, cost analysis)
    hx711                   # load-cell (weight sensor) interface

## Usage
**Access the backend dashboard**
   Open a browser and navigate to http://IP-address>:5000 to view real‑time grading cards, defect percentages.

## Project Structure
```bash
mango_grading/
├── camera_with_variety_ui.py            # UI for selecting and visualizing mango variety
├── inflation_rates_1994_2025.csv        # Historical inflation data
├── regenerate_inflation_files.py        # Script to update inflation datasets
├── train_all_varieties.py               # Train CNN models for each mango variety
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
   
