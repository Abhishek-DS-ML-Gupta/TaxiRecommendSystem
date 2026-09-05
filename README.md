[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# [A Cost-Effective Sequential Route Recommender System for Taxi Drivers](https://doi.org/10.1287/ijoc.2021.0112)

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The research data and Python scripts in this repository are a snapshot of the data and algorithms
that were used in the research reported on in the paper 
[A Cost-Effective Sequential Route Recommender System for Taxi Drivers](https://doi.org/10.1287/ijoc.2021.0112) by J. Liu, M. Teng, W. Chen, and H. Xiong. 

## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2021.0112) using its DOI and the software itself, using the following DOI.

[![DOI](https://zenodo.org/badge/597358753.svg)](https://zenodo.org/badge/latestdoi/597358753)


Below is the BibTex for citing this version of the code.

```
@article{liu2023taxi,
  author =        {Liu, Junming and Teng, Mingfei and Chen Weiwei and Xiong, Hui},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Cost-Effective Sequential Route Recommender System for Taxi Drivers},
  year =          {2023},
  doi =           {10.5281/zenodo.7631739},
  note =          {https://github.com/INFORMSJoC/2021.0112},
}  
```
## Installation and Usage

### Repository
Clone the project repository from GitHub:
```bash
git clone https://github.com/Abhishek-DS-ML-Gupta/TaxiRecommendSystem.git
cd TaxiRecommendSystem
```

### Tools and Technologies Used
* **uv**: An extremely fast Python package installer and resolver. We use `uv` for managing the virtual environment and installing dependencies.
* **FastAPI**: A modern, high-performance web framework for building the backend APIs.
* **Uvicorn**: An ASGI web server implementation used to run the FastAPI backend.
* **python-dotenv**: Used to securely load environment variables (like API keys) from a `.env` file.
* **Leaflet**: A leading open-source JavaScript library for mobile-friendly interactive maps.
* **CARTO**: A platform for spatial data analysis and visualization, used here for high-quality dark mode basemaps.

### 1. Environment Setup
We recommend using [uv](https://github.com/astral-sh/uv) to manage your Python environment.
```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate
# Or on macOS/Linux: source .venv/bin/activate

# Install the application dependencies extremely fast with uv
uv pip install -r requirements_app.txt
```

### 2. Map Configuration (CARTO API Key)
The web application uses CARTO Dark Matter basemaps. To remove the "API KEY REQUIRED" watermark, you must provide a free legacy CARTO Basemaps API key.

1. Go to [CARTO Basemaps API Key](https://carto.com/basemaps/apikey) and generate a free key.
2. Create a `.env` file in the root directory (e.g. `myroute/.env`).
3. Add the following line to your `.env` file:
   ```env
   CARTO_API_KEY=your_api_key_here
   ```
This key is securely loaded by the backend and passed to the frontend dynamically, ensuring it is not leaked in source control. Ensure `.env` remains in your `.gitignore`.

### 3. Running the Application
Once the dependencies are installed and the `.env` file is configured, start the web server using Uvicorn:
```bash
# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Then, open your web browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

## Requirements

For this project, we use the following Python Packages:

1. [**Networkx**](https://networkx.org/) for the creation and study of the structrue, dynamics, and properties of road networks.
2. Tensorflow(v2.3.0) is an end-to-end platform for machine learning. It supports deep learning model contruction, training and export.
3. Keras (v2.3.1) is a deep learning API written in Python.
4. NumPy (v1.19.5) is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
5. pandas (v1.3.5) is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
6. scikit-learn (v1.0.2) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms
7. SciPy (v1.7.3) is a free and open-source Python library used for scientific computing and technical computing. 

## Description

This paper develops a cost-effective sequential route recommender system to provide real-time routing recommendations for vacant taxis searching for the next passenger. We propose a <em>prediction-and-optimization framework</em> to recommend the searching route that maximizes the expected profit of the next successful passenger pickup based on the dynamic taxi demand-supply distribution. Specifically, this system features a deep learning-based predictor for passenger pickup probability prediction and a recursive searching algorithm that recommends the optimal searching route. The predictor integrates a Graph Convolution Network (GCN) to capture the spatial distribution and a Long Short-Term Memory (LSTM) to capture the temporal dynamics of taxi demand and supply. The recursion tree-based route optimization algorithm is proposed to recommend the optimal searching routes sequentially as route inquiries emerge in the system.

This repository includes three folders, **data**, **script**, and **results**

## Data files
The **data** folder contains the raw data used in the papre and a toy input dataset for testing. Specifically, the folder contains the following data files:

1. Static road network vertice coordinate and edges: refined_edges.csv, refined_vertice.csv
2. A sample Raw taxi trajectory data: sample_trajectory.csv. A full trajectory dataset is available here: [Taxi GPS Trajectory](https://www.dropbox.com/sh/20zfp32bf32bkuk/AACVgV8t8q5RR8vgsPpMhdABa?dl=0)
3. Real taxi search/pickup/dropoff events and earnings: pick_fair_time and searching_pick_drop
4. Weather conditions: weather.csv
5. A sample input data folder for prediciton model training and validation: sample_input

   ```python
   # To reconstruct the sample input provided with this repository, please run following code 
   sparse.save_npz('sample_last_dim_0.npz', sA)
   sparse.save_npz('sample_last_dim_1.npz', sB)
   sparse.save_npz('sample_last_dim_2.npz', sC)

   sA = sparse.load_npz('data/sample_input/sample_last_dim_0.npz')
   sB = sparse.load_npz('data/sample_input/sample_last_dim_1.npz')
   sC = sparse.load_npz('data/sample_input/sample_last_dim_2.npz')

   a = sA.toarray()
   b = sB.toarray()
   c = sC.toarray()

   m = np.concatenate((np.expand_dims(a, -1), np.expand_dims(b, -1), np.expand_dims(c, -1)), axis=-1)

   np.save('sample_input.npy', m)
   ```
6. A sample alculated road network properties for route optimization: network_properties.csv

## Script files

The **script** foler contains the core scripts used for data processing, prediction, and optimization.

1. Road segment properties prediction: gcn_lstm_split.py. The summary of model output is showin in the **result** folder.
2. Route optimization: path_searching_application.py. The sample result is shown in the **result** folder.

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/liujm8/Taxi-Route/tree/main).
