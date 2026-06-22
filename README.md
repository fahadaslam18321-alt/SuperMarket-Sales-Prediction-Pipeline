# 📊 SuperMarket Sales Analytics & Prediction Pipeline

An end-to-end Data Science project built with Python to clean, robustly preprocess, and model raw supermarket transaction data. The core focus of this project is to build a scalable and modular data cleaning pipeline that transforms noisy operational data into an ML-ready state to predict total sales revenue accurately.

---

## 🎯 Project Overview
In real-world data environments, datasets are filled with structural flaws, missing values, and logical anomalies. This project demonstrates a production-grade approach to handling these inconsistencies in a supermarket dataset and evaluates a predictive model after structural standardization.

### Key Deliverables:
- **Modular Data Preprocessing Framework** (Imputation, Normalization, Filtering)
- **Feature Engineering Pipeline** (Label Encoding & Binary Mapping)
- **Predictive Analytics Engine** (Linear Regression)

---

## 🛠️ Data Preprocessing & Cleaning Architecture

The pipeline processes raw transactional data through four strategic phases:

### Phase 1: Statistical Imputation (Missing Values)
- **Mean Imputation:** Applied to `Rating` and `Unit price` due to symmetric continuous distributions.
- **Median Imputation:** Applied to `Quantity` to prevent outlier skews.
- **Mode Imputation:** Applied to categorical dimensions like `Branch`, `City`, and `Payment` using the most frequent attributes.

### Phase 2: Structural Standardization & Casing
- **Text Normalization:** Standardized inconsistent inputs in `Customer type` (e.g., matching 'normal', 'NORMAL', and 'Normal' into a single uniform string).
- **Label Mapping:** Resolved fragmented categorical entries in `Gender` (mapping 'M', 'male', 'F', 'female' to standardized static markers: 'Male' / 'Female').
- **Type Casting & Strip:** Cleaned string currency markers (`$`) and commas from fields like `Tax 5%` and `Sales`, transfiguring them into mathematical float nodes.

### Phase 3: Outlier Treatment & De-duplication
- **Logical Constraints:** Filtered out physically impossible anomalies, such as negative purchase quantities or unit prices exceeding logical thresholds (e.g., > $1000).
- **De-duplication:** Eliminated identical row redundancies to ensure strict variance assessment without artificial weightage.

### Phase 4: Feature Engineering (ML-Readiness)
- **Label Encoding:** Converted geographical features (`City`, `Branch`) into uniform numeric keys (`City_n`, `Branch_n`) using `LabelEncoder`.
- **Binary Conversion:** Transformed the `Gender` string fields into automated binary arrays (`Gender_n` where Female = 0, Male = 1).

---

## 📈 Model Architecture & Evaluation

The predictive engine targets the total transaction revenue stream using a Linear Feature Distribution.

### Core Formulation:
$$\text{Sales} = \beta_0 + \beta_1(\text{Unit Price}) + \beta_2(\text{Quantity}) + \beta_3(\text{Gender})$$

### Execution Parameters:
- **Target Variable ($y$):** `Sales`
- **Predictor Evaluation Space ($X$):** `Unit price`, `Quantity`, `Gender_n`
- **Validation Splitting:** 80% Training / 20% Testing Framework

### Performance Metrics:
- **Coefficient of Determination ($R^2$ Score):** **0.887 (88.7%)**
- *Insight:* The model successfully explains **88.7% of the variance** in total sales revenue based entirely on the processed structural features, representing a highly reliable predictive baseline for retail logistics.

---

## 🚀 Technologies Used
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn
- **Concepts:** Linear Regression, Data Imputation, Feature Transformation, Outlier Treatment
