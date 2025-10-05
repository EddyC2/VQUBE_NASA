# 🌍 Ecosphere

#Link to Youtube Video: https://youtube.com

Ecosphere is an interactive globe that visualizes global CO₂, GHP, and PAR trends using machine learning, Python, HTML, CSS, and JavaScript. Developed for the NASA Space Apps Challenge 2025, the project explores global pollution and environmental impact using NASA open datasets from 1995 to 2020. It uses machine learning algorithms to analyze historical environmental data, predict future trends up to 2039, and cluster countries based on CO₂ emission patterns, providing insights into global environmental performance with use of kmeans cluster algorithm.

To run the project, first install all required Python libraries with pip install -r requirements.txt. Ensure the datasets (CDA_ind.csv, GHP_ind.csv, PAR_ind.csv) are present in the project directory. Start by running app.py to generate the prediction files using machine learning models, followed by Co2chart.py to execute the CO₂ trend clustering algorithm. Once these scripts have been run, open indexNASA.html in your browser to explore the interactive visualization.

The main technologies used include Python for data processing, analysis, and machine learning, HTML, CSS, and JavaScript for the web interface, and libraries such as pandas, numpy, matplotlib, scikit-learn, and plotly. Key files include the prediction and clustering scripts (app.py, Co2chart.py), datasets (CDA_ind.csv, GHP_ind.csv, PAR_ind.csv), the output clustering file (co2_trend_clusters.csv), and the main webpage (indexNASA.html).

Team members: Eduardo Diego, Yesenia Guzman, Darshana Shah, Bhavik Pandya, Jignesh Sakhia. Future improvements could include integrating live NASA API data, adding additional pollution indicators, and enhancing the 3D visualization and animations. The project is open-source for educational and environmental awareness purposes.

## 📊 Datasets

The project uses NASA open datasets and the **2020 Environmental Performance Index (EPI)** to analyze and visualize global environmental trends from 1995–2020.

 NASA datasets
- CO₂, GHP, PAR datasets in CSV format

2020 Environmental Performance Index (EPI)
- **Title:** 2020 Environmental Performance Index (EPI)  
- **Creator:** Yale Center for Environmental Law and Policy (YCELP) - Yale University, and Center for International Earth Science Information Network (CIESIN) - Columbia University  
- **Publisher:** Yale Center for Environmental Law and Policy (YCELP)/Yale University  
- **Release Date:** 2020-11-09  
- **DOI / Link:** [https://doi.org/10.7927/f54c-0r44](https://doi.org/10.7927/f54c-0r44)  

**Citation:**  
Wendling, Z.A., Emerson, J.W., de Sherbinin, A., Etsy, D.C., et al. 2020. *Environmental Performance Index 2020*. New Haven, CT: Yale Center for Environmental Law and Policy. [https://doi.org/10.13140/RG.2.2.21182.51529](https://doi.org/10.13140/RG.2.2.21182.51529)
