# Caretria Sample Dashboard

Production-grade OTC pharmacy sales analytics dashboard built with Dash.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open [http://localhost:8050](http://localhost:8050) in your browser.

## Features

- **Product Analysis** – Revenue share, top products, monthly/daily trends
- **Sales Person Performance** – Per-rep breakdown: products, revenue, countries, monthly trend
- **Country Analysis** – Regional revenue, product mix, sales person contribution
- **Product Insights** – Drill-down: avg order value gauge, country pie, revenue trend, boxes histogram
- **Statistical Analysis** – Boxplot (with/without outliers), Q-Q plot, normality tests, transformations, scatter with trend line, correlation heatmap

## Data

Uses `pharmacy_otc_sales_data.csv` (Date, Product, Sales Person, Boxes Shipped, Amount ($), Country).
