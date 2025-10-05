# 🌍 Ecosphere

**Ecosphere** is an interactive globe that visualizes global CO₂, GHP, and PAR trends using Python, HTML, CSS, and JavaScript. Developed for the **NASA Space Apps Challenge 2025**, the project explores global pollution and environmental impact using NASA open datasets from 1995 to 2020. It demonstrates how environmental factors have changed over time and predicts future trends up to 2039.

To run the project, first install all required Python libraries with `pip install -r requirements.txt`. Ensure the datasets (`CDA_ind.csv`, `GHP_ind.csv`, `PAR_ind.csv`) are present in the project directory. Start by running `app.py` to generate the prediction files, followed by `Co2chart.py` to execute the CO₂ trend clustering algorithm. Once these scripts have been run, open `indexNASA.html` in your browser to explore the interactive visualization.

The main technologies used include Python for data processing and analysis, HTML, CSS, and JavaScript for the web interface, and libraries such as `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `plotly`. Key files include the prediction and clustering scripts (`app.py`, `Co2chart.py`), datasets (`CDA_ind.csv`, `GHP_ind.csv`, `PAR_ind.csv`), the output clustering file (`co2_trend_clusters.csv`), and the main webpage (`indexNASA.html`).  

Team members: Eduardo Diego, Yesenia Guzman, Darshana Shah, Bhavik Pandya, Jignesh Sakhia. Future improvements could include integrating live NASA API data, adding additional pollution indicators, and enhancing the 3D visualization and animations. The project is open-source for educational and environmental awareness purposes.
