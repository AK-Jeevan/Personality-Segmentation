# 🧠 Personality-Based Segmentation Using DBSCAN

This project segments individuals based on their personality traits using unsupervised learning. By applying clustering techniques to synthetic personality data, we uncover distinct behavioral groups.

## 📦 Dataset

- `personality_synthetic_dataset.csv`: Contains synthetic personality trait scores for individuals.

## 🧪 Methodology

1. **Preprocessing**
   - Standardized features using `StandardScaler` to normalize trait scores.

2. **Dimensionality Reduction**
   - Applied `PCA` to reduce dimensionality and visualize clusters.

3. **Clustering**
   - Used `DBSCAN` to identify personality-based clusters.
   - DBSCAN is ideal for discovering non-linear patterns and handling noise.

4. **Evaluation**
   - Used `Silhouette Score` to assess clustering quality and tune DBSCAN parameters.

## 📈 Results

- Individuals grouped into distinct personality clusters.
- PCA visualization revealed meaningful separations.
- Silhouette Score guided optimal DBSCAN configuration.

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## 📁 Repository Structure

├── personality_synthetic_dataset.csv # Dataset 
├── Personality_Segmentation.py # Main script for preprocessing, PCA, DBSCAN, and visualization
├── README.md # Project documentation


## 📜 License

This project is licensed under the MIT License.
