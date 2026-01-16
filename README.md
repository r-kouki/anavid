# Student Employability Prediction

Machine learning project predicting student employability using Random Forest classification with 92% accuracy. Includes an interactive Streamlit web app and comprehensive Quarto presentation.

## 🌐 Live Demos

- **🎈 Streamlit App:** [https://anavid-kouki.streamlit.app/](https://anavid-kouki.streamlit.app/) - Interactive predictions
- **📊 Presentation:** [https://USERNAME.github.io/REPO-NAME/](https://USERNAME.github.io/REPO-NAME/) - Technical presentation (update after GitHub deployment)

## 📋 Project Overview

Binary classification model predicting student employability with:
- 🤖 Random Forest classifier (92% accuracy)
- 🌐 Streamlit web app (manual entry + Excel upload)
- 📊 Quarto slides (technical + non-technical explanations)
- 📈 37 features from 260 student records

## 🚀 Quick Start

### Option 1: Try the Deployed App
Visit the live Streamlit app (see deployment instructions below) to:
- EnRun Locally

```bash
# Use the quick start script
./run.sh

# Or manually:
source assignment/bin/activate  # Linux/Mac
pip install -r requirements.txt
python classification_model.py  # Train model
streamlit run app.py            # Launch app
```

Visit `http://localhost:8501` in your browser.

```
assignement  (Copy)/
├── classification_model.py   # Model training script
├── app.py                     # Streamlit web application
├── presentation.qmd           # Quarto slides source
├── data.xlsx                  # Training dataset (260 students)
├── requirements.txt           # Python dependencies
├── _quarto.yml               # Quarto configuration
├── custom.scss               # Presentation styling
├── styles.css                # Additional CSS
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── .github/
│   └── workflows/
│       └── quarto-publish.yml # GitHub Actions for auto-deployment
├── model.pkl                 # Saved trained model
├── imputer.pkl              # Saved preprocessor
├── feature_names.pkl        # Feature list
└── docs/                    # Generated Quarto output (GitHub Pages)
```

## 📊 Data

- **File:** `data.xlsx`
- **Size:** 260 students, 39 columns
- **Target:** Binary classification (Class 0 = Not Highly Employable, Class 1 = Highly Employable)
- **Features:** 37 features including:
  - Demographics (Gender, Nationality, Major, Level)
  - Innovation & Entrepreneurship scores (IE1-IE5)
  - Soft Skills scores (SMSK1-SMSK4)
  - Research & Analytical Skills (RAS1-RAS5)
  - Technical & Leadership (TL1-TL3)
  - Professional Skills Development (PSD1-PSD5)
  - Industry Metrics (IM1-IM6)
  - Work Experience (W1-W3)
  - Employment Status & Overall Score

**Class Balance:** Almost perfectly balanced (133 Class 0, 127 Class 1) ✅

## 🤖 Model Pipeline

### 1. Data Loading
- Reads Excel file directly using pandas and openpyxl

├── classification_model.py   # Model training script
├── app.py                     # Streamlit web application
├── presentation.qmd           # Quarto slides
├── data.xlsx                  # Dataset (260 students, 37 features)
├── requirements.txt           # Dependencies
├── run.sh                     # Quick start script
├── model.pkl                 # Trained model
├── imputer.pkl              # Preprocessor
├── feature_names.pkl        # Feature list
└── .github/workflows/        # GitHub Actions for deployment
## 📈 Results

### Cross-Validation Performance
- **Mean F1 Score:** 0.9683 (±0.065)
- Consistent performance across all 5 folds

### Test Set Performance
- **Overall Accuracy:** 92%
- **Class 0:** F1-score 0.93 (Precision: 0.90, Recall: 0.96)
- **Class 1:** F1-score 0.92 (Precision: 0.96, Recall: 0.88)

## 📊 Visualizations

The script generates `model_results.png` with two plots:

![Model Results](model_results.png)

### Confusion Matrix
Shows the distribution of correct and incorrect predictions for each class.

### Top 10 Feature Importances
Identifies which features the Random Forest model relies on most for making predictions.

---

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app and model"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important Files:**
   - `requirements.txt` - Auto-installed by Streamlit Cloud
   - `.streamlit/config.toml` - App configuration
   - Model artifacts (`*.pkl`) must be committed to repo

**Note:** Make sure to include model files in your repository by temporarily removing them from `.gitignore`:
```bash
git add -f model.pkl imputer.pkl feature_names.pkl
git commit -m "Add model artifacts for deployment"
```

### GitHub Pages Deployment

The Quarto presentation automatically deploys to GitHub Pages via GitHub Actions.

1. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` / root
   - Save

2. **Push Changes:**
   ```bash
   git add .
   git commit -m "Add Quarto presentation"
   git push origin main
   ```

3. **Automatic Deployment:**
   - GitHub Actions workflow (`.github/workflows/quarto-publish.yml`) runs automatically
## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub: `git push origin main`
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy `app.py`
4. Ensure model files (*.pkl) are committed: `git add -f *.pkl`

### GitHub Pages
1. Enable Pages in repo Settings → Pages → `gh-pages` branch
2. Push code - GitHub Actions auto-deploys presentation
3. Visit: `https://[username].github.io/[repo-name]/presentation.html`

### Local Quarto Preview
```bash
quarto render presentation.qmd  # Generate slides
quarto preview presentation.qmd # Live preview
```

## 🎯 Using the App

**Manual Entry:** Fill 37 fields → Click "Load Example Data" for demo → Predict

**Excel Upload:** Upload .xlsx file → Batch predictions → Download CSV results

**File Format:** Must have all 37 features (see data.xlsx as template) questions or feedback, please open an issue on GitHub.

## 📄 License

This project is for educational purposes.

---

**Built with ❤️ using Python, scikit-learn, Streamlit, and Quarto**
Files Generated

- `model.pkl`, `imputer.pkl`, `feature_names.pkl` - Model artifacts
- `model_results.png` - Performance visualizations
- `docs/` - Rendered presentation (GitHub Pages)

---

**Built with Python • scikit-learn • Streamlit •