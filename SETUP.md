# Setup Instructions

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation Steps

1. **Navigate to the project directory:**
   ```bash
   cd yelp-review-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download your Yelp dataset:**
   - Get the Yelp Academic Dataset from [Yelp's website](https://www.yelp.com/dataset)
   - Extract the `yelp_academic_dataset_review.json` file
   - Place it in a location you can access

## Running the Analysis

### Option 1: Jupyter Notebook (Recommended)

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Click on `yelp_analysis.ipynb`

3. **Update the data path:**
   - In the "Load Yelp Dataset" cell, update `DATA_FILE` with your file path
   - Example: `DATA_FILE = r"C:\Users\YourName\Downloads\yelp_academic_dataset_review.json"`

4. **Run all cells:**
   - Go to Cell → Run All
   - Or run cells individually with Shift + Enter

### Option 2: Command Line

```bash
python main_analysis.py --data path/to/your/reviews.json --format json --sample 5000
```

**Command line options:**
- `--data`: Path to your Yelp review JSON file (required)
- `--format`: Data format - `json` or `csv` (default: `json`)
- `--sample`: Number of reviews to analyze (optional, for faster testing)
- `--text-column`: Column name containing review text (default: `text`)
- `--output`: Output directory for visualizations (default: `output`)

### Option 3: Windows Scripts

- **Batch file:** Double-click `run_analysis.bat` (update the file path inside first)
- **PowerShell:** Right-click `run_analysis.ps1` → Run with PowerShell

## First Run Notes

- NLTK data will be downloaded automatically (may take a few minutes)
- The first analysis may be slower as models initialize
- For large datasets, start with a small sample (e.g., 5000 reviews)

## Troubleshooting

**Python not found:**
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

**NLTK download errors:**
- The notebook will automatically retry downloads
- If issues persist, run: `python -c "import nltk; nltk.download('all')"`

**File not found:**
- Use absolute paths (full path from root)
- Or move your data file to the project directory

## Output

Results are saved to:
- **Notebook:** Visualizations display inline
- **Command line:** Saved to `output/` directory
  - Charts (PNG files)
  - Results JSON file

