Week 1: Advanced EDA & Feature Engineering

DecodeLabs Data Science Internship — Project 1

Overview

This project focuses on transforming a raw, messy retail sales dataset into a clean, mathematically sound dataset ready for machine learning. The core goal wasn't just to "clean the data" — it was to diagnose why each issue existed and choose the most appropriate, justified technique to fix it.

Dataset
Source: Retail Store Sales dataset (Kaggle)
Size: 12,575 rows × 11 columns
Columns: Transaction ID, Customer ID, Category, Item, Price Per Unit, Quantity, Total Spent, Payment Method, Location, Transaction Date, Discount Applied

Objectives
Handle missing data using statistical imputation and logical recovery methods
Identify and neutralize outliers using the IQR method
Engineer at least 3 new predictive features

Phase 1: Handling Missing Data

Rather than applying a single blanket method (e.g. mean imputation) across the board, each column's missingness was diagnosed individually:

Column	Missing %	Method Used	Reasoning
Discount Applied	~33.4%	Filled with "Unknown" category	True/False/NaN were roughly evenly split — missingness looked random, not a data error. Dropping ~4,200 rows would have meant significant data loss, so missingness was treated as its own valid state.
Price Per Unit	~4.8%	Formula recovery: Price = Total Spent ÷ Quantity	All 609 missing rows had both Total Spent and Quantity present, allowing exact mathematical recovery instead of statistical guessing.
Quantity & Total Spent	~4.8% (missing together)	Median imputation	Investigated missingness patterns first — found these two always went missing together, so the price formula couldn't be used. Checked distribution shape via histograms: Total Spent was right-skewed, so median was used over mean to avoid distortion from extreme values.
Item	~9.65%	Category–Price lookup mapping	Verified that each (Category, Price Per Unit) combination maps to exactly one unique Item (checked via groupby().nunique()). Built a lookup table from complete records and used it to recover all 1,213 missing values with zero guessing.

Result: 0 missing values across the entire dataset.

Phase 2: Outlier Detection & Treatment (IQR Method)

Applied the Interquartile Range (IQR) method to all three numeric columns:

IQR = Q3 − Q1
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Column	Outliers Found	Action Taken
Total Spent	157 rows above upper bound (377.5)	Capped (Winsorized) to the upper bound instead of deleting rows — preserving row count and data volume while neutralizing the extreme values' influence
Quantity	0	No treatment needed
Price Per Unit	0	No treatment needed

Why capping over deletion? The outlier values in Total Spent were legitimate, mathematically consistent transactions (Price × Quantity checked out) — not data errors. Deleting them would discard real information, so values were capped at the statistical boundary instead.

Phase 3: Feature Engineering

Three new predictive features were engineered from the existing columns:

Transaction Month — extracted from Transaction Date to capture seasonal spending patterns
Transaction Day of Week — extracted from Transaction Date to capture weekday vs. weekend behavior
Discount Encoded — numeric encoding of Discount Applied (True → 1, False → 0, Unknown → -1) to make the feature model-ready

An additional EDA finding: grouping Total Spent by Discount Applied showed no significant difference in average spending across True/False/Unknown groups — suggesting discounts in this dataset are applied independent of purchase value.

Tools & Libraries
pandas, numpy — data manipulation
matplotlib, seaborn — distribution visualization
scipy.stats — statistical functions
scikit-learn — imputation utilities

Files
Week_1_Advanced_EDA_Feature_Engineering.ipynb — full notebook
retail_store_sales.csv — raw dataset

Key Takeaway

Data cleaning isn't a mechanical checklist — every missing value or outlier tells a story about why it exists, and that story should drive the fix. Formula-based recovery and lookup mapping recovered data with certainty where blind statistical imputation would have only guessed.

Part of the DecodeLabs Data Science Internship — Industrial Training Kit, Batch 2026
