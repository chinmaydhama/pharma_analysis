"""
Caretria Sample Dashboard
Production-grade OTC Pharmacy Sales Analytics
"""

import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pharmacy_otc_sales_data.csv")


def load_data():
    """Load and preprocess pharmacy OTC sales data."""
    df = pd.read_csv(DATA_PATH)
    # Normalize column names
    df.columns = [c.replace("Amount ($)", "Amount").strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayName"] = df["Date"].dt.strftime("%a")
    df["Year"] = df["Date"].dt.year
    return df


df_raw = load_data()
df = df_raw.copy()

# Precompute KPIs
total_revenue = df["Amount"].sum()
total_boxes = df["Boxes Shipped"].sum()
unique_products = df["Product"].nunique()
unique_countries = df["Country"].nunique()
unique_sales_people = df["Sales Person"].nunique()
total_transactions = len(df)
avg_order_value = round(total_revenue / total_transactions, 2)

# Caretria brand colors
CARETRIA_TEAL = "#0D9488"
CARETRIA_EMERALD = "#10B981"
CARETRIA_DARK = "#0F766E"
CARETRIA_LIGHT = "#5EEAD4"
CARETRIA_ACCENT = "#F59E0B"
CARETRIA_MUTED = "#64748b"

PLOT_TEMPLATE = dict(
    layout=dict(
        font=dict(family="DM Sans, system-ui, sans-serif", size=12, color="#0f172a"),
        title=dict(font=dict(size=16, color="#0f172a"), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        margin=dict(t=56, b=48, l=56, r=32),
        hoverlabel=dict(bgcolor="#fff", font_size=12, font_family="DM Sans"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
        colorway=[CARETRIA_TEAL, CARETRIA_EMERALD, CARETRIA_DARK, CARETRIA_LIGHT, CARETRIA_ACCENT, "#64748b"],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
)


def apply_theme(fig, height=None):
    """Apply production-grade theme to a Plotly figure."""
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    if height:
        fig.update_layout(height=height)
    try:
        fig.update_traces(marker=dict(line=dict(width=0)), selector=dict(type="bar"))
    except Exception:
        pass
    return fig


# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=["https://use.fontawesome.com/releases/v5.8.1/css/all.css"],
    title="Caretria Sample Dashboard",
)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

KPI_VALUES = [total_revenue, total_boxes, unique_products, unique_countries, unique_sales_people, avg_order_value]
KPI_LABELS = [
    f"${total_revenue:,.0f}",
    f"{total_boxes:,}",
    str(unique_products),
    str(unique_countries),
    str(unique_sales_people),
    f"${avg_order_value}",
]
KPI_DESCRIPTIONS = [
    "Total Revenue",
    "Total Boxes Shipped",
    "Unique Products",
    "Countries",
    "Sales People",
    "Avg Order Value",
]
KPI_ICONS = ["fa-dollar-sign", "fa-box-open", "fa-pills", "fa-globe", "fa-users", "fa-chart-line"]

app.layout = html.Div(
    id="main-container",
    children=[
        html.Header(
            className="dash-header",
            children=[
                html.H1("Caretria Sample Dashboard"),
                html.P(
                    "OTC Pharmacy sales analytics: products, sales performance, regional insights & statistics"
                ),
            ],
        ),
        html.Div(
            className="kpi-grid",
            children=[
                html.Div(
                    className="kpi-card",
                    children=[
                        html.I(className=f"fas {icon} kpi-icon"),
                        html.Div(str(value), className="kpi-value"),
                        html.Div(description, className="kpi-label"),
                    ],
                )
                for icon, value, description in zip(KPI_ICONS, KPI_LABELS, KPI_DESCRIPTIONS)
            ],
        ),
        html.Div(
            className="Tab-container",
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    children=[
                        dcc.Tab(label="Product Analysis", value="tab-1"),
                        dcc.Tab(label="Sales Person Performance", value="tab-2"),
                        dcc.Tab(label="Country Analysis", value="tab-3"),
                        dcc.Tab(label="Product Insights", value="tab-4"),
                        dcc.Tab(label="Statistical Analysis", value="tab-5"),
                    ],
                ),
                html.Div(id="tabs-content", className="tab-content-inner"),
            ],
        ),
    ],
)

# -----------------------------------------------------------------------------
# Tab Content Renderer
# -----------------------------------------------------------------------------


@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab_content(tab):
    if tab == "tab-1":
        return html.Div(
            [
                html.Div(
                    className="control-panel",
                    children=[
                        html.Div(
                            [
                                html.Label("Product"),
                                dcc.Dropdown(
                                    id="product-dropdown",
                                    options=[{"label": p, "value": p} for p in df["Product"].unique()],
                                    value=df["Product"].unique()[0],
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Top N"),
                                dcc.RadioItems(
                                    id="product-top-n-radio",
                                    options=[{"label": "Top 5", "value": 5}, {"label": "Top 10", "value": 10}],
                                    value=7,
                                    inline=True,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="metric-strip",
                    children=[
                        html.Div(id="product-total-revenue", className="metric-box"),
                        html.Div(id="product-total-boxes", className="metric-box"),
                        html.Div(id="product-transactions", className="metric-box"),
                    ],
                ),
                html.Div(
                    className="chart-card",
                    children=[
                        dcc.Graph(id="product-subplots", config={"displayModeBar": True, "displaylogo": False})
                    ],
                ),
            ]
        )
    elif tab == "tab-2":
        return html.Div(
            [
                html.Div(
                    className="control-panel",
                    children=[
                        html.Div(
                            [
                                html.Label("Sales Person"),
                                dcc.Dropdown(
                                    id="salesperson-dropdown",
                                    options=[{"label": p, "value": p} for p in df["Sales Person"].unique()],
                                    value=df["Sales Person"].unique()[0],
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Show"),
                                dcc.RadioItems(
                                    id="salesperson-top-n-radio",
                                    options=[
                                        {"label": "Top 5 Products", "value": 5},
                                        {"label": "Top 10 Products", "value": 10},
                                        {"label": "All", "value": 20},
                                    ],
                                    value=5,
                                    inline=True,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="chart-card",
                    children=[
                        dcc.Graph(id="salesperson-graph", config={"displayModeBar": True, "displaylogo": False})
                    ],
                ),
            ]
        )
    elif tab == "tab-3":
        return html.Div(
            [
                html.Div(
                    className="control-panel",
                    children=[
                        html.Div(
                            [
                                html.Label("Country"),
                                dcc.Dropdown(
                                    id="country-dropdown",
                                    options=[{"label": c, "value": c} for c in df["Country"].unique()],
                                    value=df["Country"].unique()[0],
                                    clearable=False,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="metric-strip",
                    children=[
                        html.Div(id="country-revenue", className="metric-box"),
                        html.Div(id="country-boxes", className="metric-box"),
                        html.Div(id="country-products", className="metric-box"),
                    ],
                ),
                html.Div(
                    className="chart-card",
                    children=[
                        dcc.Graph(id="country-graph", config={"displayModeBar": True, "displaylogo": False})
                    ],
                ),
            ]
        )
    elif tab == "tab-4":
        return html.Div(
            [
                html.H2("Product Insights", className="section-title"),
                html.Div(
                    className="control-panel",
                    children=[
                        html.Div(
                            [
                                html.Label("Product"),
                                dcc.Dropdown(
                                    id="insight-product-selector",
                                    options=[{"label": p, "value": p} for p in df["Product"].unique()],
                                    value=df["Product"].unique()[0],
                                    clearable=False,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="chart-row",
                    children=[
                        html.Div(
                            className="chart-card",
                            children=[
                                dcc.Graph(
                                    id="insight-gauge",
                                    config={"displayModeBar": True, "displaylogo": False},
                                )
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            children=[
                                dcc.Graph(
                                    id="insight-country-pie",
                                    config={"displayModeBar": True, "displaylogo": False},
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="chart-row",
                    children=[
                        html.Div(
                            className="chart-card",
                            children=[
                                dcc.Graph(
                                    id="insight-revenue-trend",
                                    config={"displayModeBar": True, "displaylogo": False},
                                )
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            children=[
                                dcc.Graph(
                                    id="insight-boxes-histogram",
                                    config={"displayModeBar": True, "displaylogo": False},
                                )
                            ],
                        ),
                    ],
                ),
            ]
        )
    elif tab == "tab-5":
        numeric_options = [{"label": c, "value": c} for c in ["Boxes Shipped", "Amount"]]
        return html.Div(
            [
                html.H2("Statistical Analysis", className="section-title"),
                html.Div(
                    className="chart-row",
                    children=[
                        html.Div(
                            className="chart-card",
                            style={"flex": "1 1 33%"},
                            children=[
                                html.Label("Feature (boxplot)", style={"display": "block", "marginBottom": "8px"}),
                                dcc.Dropdown(
                                    id="stat-feature-dropdown",
                                    options=numeric_options,
                                    value="Amount",
                                    clearable=False,
                                ),
                                dcc.Graph(
                                    id="stat-boxplot",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            style={"flex": "1 1 33%"},
                            children=[
                                html.Label(
                                    "Normality: column",
                                    style={"display": "block", "marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="normality-column",
                                    options=numeric_options,
                                    value="Amount",
                                    clearable=False,
                                ),
                                dcc.Dropdown(
                                    id="normality-test",
                                    options=[
                                        {"label": "Shapiro-Wilk", "value": "shapiro"},
                                        {"label": "Kolmogorov-Smirnov", "value": "ks"},
                                        {"label": "D'Agostino K²", "value": "dagostino"},
                                    ],
                                    value="shapiro",
                                    clearable=False,
                                ),
                                dcc.Graph(
                                    id="stat-qq-plot",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                                html.Div(
                                    id="normality-test-result",
                                    style={
                                        "fontSize": "14px",
                                        "marginTop": "8px",
                                        "color": "#64748b",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            style={"flex": "1 1 33%"},
                            children=[
                                html.Label(
                                    "Transformation",
                                    style={"display": "block", "marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="transformation-feature",
                                    options=numeric_options,
                                    value="Amount",
                                    clearable=False,
                                ),
                                dcc.RadioItems(
                                    id="transformation-type",
                                    options=[
                                        {"label": "Log", "value": "log"},
                                        {"label": "Square root", "value": "sqrt"},
                                    ],
                                    value="log",
                                    inline=True,
                                ),
                                dcc.Graph(
                                    id="stat-transformed",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="chart-row",
                    children=[
                        html.Div(
                            className="chart-card",
                            style={"flex": "1 1 50%"},
                            children=[
                                html.Label(
                                    "Scatter: X / Y",
                                    style={"display": "block", "marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="scatter-x",
                                    options=numeric_options + [{"label": "Product (count)", "value": "_product_count"}],
                                    value="Boxes Shipped",
                                    clearable=False,
                                ),
                                dcc.Dropdown(
                                    id="scatter-y",
                                    options=numeric_options,
                                    value="Amount",
                                    clearable=False,
                                ),
                                dcc.Checklist(
                                    id="scatter-trendline",
                                    options=[{"label": "Show trend line", "value": "show"}],
                                    value=[],
                                ),
                                dcc.Graph(
                                    id="stat-scatter",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                                html.Div(
                                    id="r-squared-value",
                                    style={
                                        "fontSize": "14px",
                                        "marginTop": "8px",
                                        "color": "#64748b",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            style={"flex": "1 1 50%"},
                            children=[
                                html.Label(
                                    "Correlation Heatmap",
                                    style={"display": "block", "marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="stat-correlation",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
    return html.Div()


# -----------------------------------------------------------------------------
# Tab 1: Product Analysis Callbacks
# -----------------------------------------------------------------------------


@app.callback(
    [
        Output("product-total-revenue", "children"),
        Output("product-total-boxes", "children"),
        Output("product-transactions", "children"),
    ],
    Input("product-dropdown", "value"),
)
def update_product_metrics(product):
    if not product:
        raise PreventUpdate
    f = df[df["Product"] == product]
    rev = f["Amount"].sum()
    boxes = f["Boxes Shipped"].sum()
    n = len(f)
    return (
        f"Revenue: ${rev:,.0f}",
        f"Boxes: {boxes:,}",
        f"Transactions: {n}",
    )


@app.callback(
    Output("product-subplots", "figure"),
    [Input("product-dropdown", "value"), Input("product-top-n-radio", "value")],
)
def update_product_figures(product, top_n):
    top_n = min(top_n or 7, df["Product"].nunique())
    top_products = df.groupby("Product").agg({"Amount": "sum", "Boxes Shipped": "sum"}).reset_index()
    top_products = top_products.nlargest(top_n, "Amount")["Product"].tolist()
    f = df[df["Product"].isin(top_products)]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "Revenue share (top products)",
            "Revenue by product",
            "Monthly revenue trend",
            "Revenue by day of week",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    rev_by_product = df[df["Product"].isin(top_products)].groupby("Product")["Amount"].sum().sort_values(ascending=True)
    colors = [CARETRIA_TEAL, CARETRIA_EMERALD, CARETRIA_DARK, CARETRIA_LIGHT, CARETRIA_ACCENT, "#94a3b8", "#cbd5e1"][
        :top_n
    ]
    fig.add_trace(
        go.Pie(
            labels=rev_by_product.index,
            values=rev_by_product.values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=rev_by_product.index,
            x=rev_by_product.values,
            orientation="h",
            marker_color=CARETRIA_TEAL,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    monthly = f.groupby("Month")["Amount"].sum()
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly.values,
            mode="lines+markers",
            line=dict(color=CARETRIA_TEAL, width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    dow_order = [1, 2, 3, 4, 5, 6, 0]  # Mon-Sun
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = f.groupby("DayOfWeek")["Amount"].sum().reindex(dow_order, fill_value=0)
    fig.add_trace(
        go.Bar(
            x=dow_labels,
            y=[by_dow.get(i, 0) for i in range(7)],
            marker_color=CARETRIA_ACCENT,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=2)
    fig.update_yaxes(title_text="Revenue ($)", row=2, col=2)
    fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    apply_theme(fig, height=650)
    return fig


# -----------------------------------------------------------------------------
# Tab 2: Sales Person Callbacks
# -----------------------------------------------------------------------------


@app.callback(
    Output("salesperson-graph", "figure"),
    [Input("salesperson-dropdown", "value"), Input("salesperson-top-n-radio", "value")],
)
def update_salesperson_graph(salesperson, top_n):
    f = df[df["Sales Person"] == salesperson]
    n = min(top_n or 5, f["Product"].nunique())

    order_counts = f["Product"].value_counts().head(n)
    rev_by_product = f.groupby("Product")["Amount"].sum().nlargest(n)
    boxes_by_country = f.groupby("Country")["Boxes Shipped"].sum()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top products by transactions",
            "Top products by revenue",
            "Boxes shipped by country",
            "Monthly revenue",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=order_counts.values,
            y=order_counts.index,
            orientation="h",
            marker_color=CARETRIA_TEAL,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=rev_by_product.values,
            y=rev_by_product.index,
            orientation="h",
            marker_color=CARETRIA_EMERALD,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=boxes_by_country.index,
            y=boxes_by_country.values,
            marker_color=CARETRIA_ACCENT,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    monthly = f.groupby("Month")["Amount"].sum()
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly.values,
            mode="lines+markers",
            line=dict(color=CARETRIA_DARK, width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Transactions", row=1, col=1)
    fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    fig.update_xaxes(title_text="Country", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    apply_theme(fig, height=700)
    return fig


# -----------------------------------------------------------------------------
# Tab 3: Country Callbacks
# -----------------------------------------------------------------------------


@app.callback(
    [
        Output("country-revenue", "children"),
        Output("country-boxes", "children"),
        Output("country-products", "children"),
    ],
    Input("country-dropdown", "value"),
)
def update_country_metrics(country):
    if not country:
        raise PreventUpdate
    f = df[df["Country"] == country]
    rev = f["Amount"].sum()
    boxes = f["Boxes Shipped"].sum()
    prods = f["Product"].nunique()
    return (
        f"Revenue: ${rev:,.0f}",
        f"Boxes: {boxes:,}",
        f"Products: {prods}",
    )


@app.callback(Output("country-graph", "figure"), [Input("country-dropdown", "value")])
def update_country_graph(country):
    f = df[df["Country"] == country]

    rev_by_product = f.groupby("Product")["Amount"].sum().sort_values(ascending=True)
    sales_by_person = f.groupby("Sales Person")["Amount"].sum().sort_values(ascending=True)
    monthly = f.groupby("Month")["Amount"].sum()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Revenue by product",
            "Revenue by sales person",
            "Monthly trend",
            "Product mix (pie)",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            y=rev_by_product.index,
            x=rev_by_product.values,
            orientation="h",
            marker_color=CARETRIA_TEAL,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=sales_by_person.index,
            x=sales_by_person.values,
            orientation="h",
            marker_color=CARETRIA_EMERALD,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly.values,
            mode="lines+markers",
            line=dict(color=CARETRIA_DARK, width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=rev_by_product.index,
            values=rev_by_product.values,
            hole=0.5,
            marker=dict(
                colors=[CARETRIA_TEAL, CARETRIA_EMERALD, CARETRIA_DARK, CARETRIA_LIGHT, CARETRIA_ACCENT, "#94a3b8", "#cbd5e1"]
            ),
            textinfo="percent",
            textposition="inside",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    apply_theme(fig, height=700)
    return fig


# -----------------------------------------------------------------------------
# Tab 4: Product Insights Callbacks
# -----------------------------------------------------------------------------


@app.callback(
    [
        Output("insight-gauge", "figure"),
        Output("insight-country-pie", "figure"),
        Output("insight-revenue-trend", "figure"),
        Output("insight-boxes-histogram", "figure"),
    ],
    [Input("tabs", "value"), Input("insight-product-selector", "value")],
)
def update_product_insights(tab_value, product):
    if tab_value != "tab-4":
        raise PreventUpdate
    if not product or product not in df["Product"].unique():
        product = df["Product"].unique()[0]
    f = df[df["Product"] == product]

    avg_amount = f["Amount"].mean()
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(avg_amount, 2),
            number=dict(prefix="$", font=dict(size=28)),
            domain={"x": [0.1, 0.9], "y": [0.15, 0.85]},
            title={"text": "Avg Order Value", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, f["Amount"].max() * 1.1], "tickwidth": 1},
                "bar": {"color": CARETRIA_TEAL},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, avg_amount * 0.33], "color": "#f1f5f9"},
                    {"range": [avg_amount * 0.33, avg_amount * 0.66], "color": "#e2e8f0"},
                    {"range": [avg_amount * 0.66, f["Amount"].max() * 1.1], "color": "#cbd5e1"},
                ],
                "threshold": {
                    "line": {"color": CARETRIA_TEAL, "width": 4},
                    "thickness": 0.8,
                    "value": avg_amount,
                },
            },
        )
    )
    apply_theme(gauge_fig, height=280)

    by_country = f.groupby("Country")["Amount"].sum()
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=by_country.index,
                values=by_country.values,
                hole=0.5,
                marker=dict(
                    colors=[CARETRIA_TEAL, CARETRIA_EMERALD, CARETRIA_DARK, CARETRIA_LIGHT, CARETRIA_ACCENT]
                ),
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                hovertemplate="%{label}<br>Revenue: $%{value:,.0f}<extra></extra>",
            )
        ]
    )
    pie_fig.update_layout(
        title_text="Revenue by Country",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05),
    )
    apply_theme(pie_fig, height=280)

    monthly = f.groupby("Month")["Amount"].sum()
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly.values,
            fill="tozeroy",
            line=dict(color=CARETRIA_TEAL, width=2),
            fillcolor="rgba(13, 148, 136, 0.15)",
            name="Revenue",
        )
    )
    trend_fig.update_layout(
        title_text="Revenue by Month",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
    )
    apply_theme(trend_fig, height=280)

    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=f["Boxes Shipped"],
            nbinsx=min(20, max(5, f["Boxes Shipped"].nunique())),
            marker_color=CARETRIA_EMERALD,
        )
    )
    hist_fig.update_layout(
        title_text="Boxes Shipped Distribution",
        xaxis_title="Boxes",
        yaxis_title="Count",
    )
    apply_theme(hist_fig, height=280)

    return gauge_fig, pie_fig, trend_fig, hist_fig


# -----------------------------------------------------------------------------
# Tab 5: Statistical Analysis Callbacks
# -----------------------------------------------------------------------------

# Boxplot
@app.callback(Output("stat-boxplot", "figure"), Input("stat-feature-dropdown", "value"))
def update_boxplot(feature):
    data_col = df[feature].dropna()
    Q1 = data_col.quantile(0.25)
    Q3 = data_col.quantile(0.75)
    IQR = Q3 - Q1
    filtered = df[(df[feature] >= Q1 - 1.5 * IQR) & (df[feature] <= Q3 + 1.5 * IQR)][feature]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("With outliers", "IQR filtered"),
    )
    fig.add_trace(
        go.Box(y=df[feature], name="With outliers", marker_color=CARETRIA_TEAL, line_color=CARETRIA_DARK),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(y=filtered, name="Filtered", marker_color=CARETRIA_ACCENT, line_color=CARETRIA_DARK),
        row=1,
        col=2,
    )
    fig.update_layout(showlegend=False)
    apply_theme(fig, height=340)
    return fig


# QQ plot and normality
@app.callback(
    [Output("stat-qq-plot", "figure"), Output("normality-test-result", "children")],
    [
        Input("normality-column", "value"),
        Input("normality-test", "value"),
    ],
)
def update_qq_and_test(column, test_type):
    sample = df[column].dropna()
    n = min(1000, len(sample))
    data = sample.sample(n=n, replace=False, random_state=42)

    if test_type == "shapiro":
        stat, p = shapiro(data)
        test_name = "Shapiro-Wilk"
    elif test_type == "ks":
        stat, p = kstest(data, "norm", args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"
    else:
        stat, p = normaltest(data)
        test_name = "D'Agostino K²"

    (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr * slope + intercept,
            mode="lines",
            name="Theoretical",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Data",
            marker=dict(color=CARETRIA_TEAL, size=6, line=dict(width=0)),
        )
    )
    fig.update_layout(
        title_text=f"Q-Q plot: {column}",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Sample quantiles",
    )
    apply_theme(fig, height=300)
    result_text = f"{test_name}: stat = {stat:.3f}, p = {p:.4f}"
    return fig, result_text


# Transformation
@app.callback(
    Output("stat-transformed", "figure"),
    [Input("transformation-feature", "value"), Input("transformation-type", "value")],
)
def update_transformed(feature, transform_type):
    raw = df[feature].dropna()
    if transform_type == "log":
        transformed = np.log1p(raw)
        title = f"Log(1 + {feature})"
    else:
        transformed = np.sqrt(raw)
        title = f"√{feature}"
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=transformed,
            nbinsx=min(40, max(15, int(transformed.nunique() / 2))),
            marker_color=CARETRIA_TEAL,
        )
    )
    fig.update_layout(title_text=title, xaxis_title="Value", yaxis_title="Count")
    apply_theme(fig, height=300)
    return fig


# Scatter and correlation
numeric_cols = ["Boxes Shipped", "Amount"]
corr_matrix = df[numeric_cols].corr()


@app.callback(
    [Output("stat-scatter", "figure"), Output("r-squared-value", "children")],
    [
        Input("scatter-x", "value"),
        Input("scatter-y", "value"),
        Input("scatter-trendline", "value"),
    ],
)
def update_scatter(x_col, y_col, trendline_val):
    show_trend = "show" in (trendline_val or [])
    sample_df = df.copy()
    if x_col == "_product_count":
        sample_df["_product_count"] = sample_df.groupby("Product")["Product"].transform("count")
        x_col_use = "_product_count"
    else:
        x_col_use = x_col

    fig = px.scatter(
        data_frame=sample_df,
        x=x_col_use,
        y=y_col,
        opacity=0.6,
        trendline="ols" if show_trend else None,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)), selector=dict(mode="markers"))
    fig.update_layout(
        title_text=f"{x_col_use} vs {y_col}",
        xaxis_title=x_col_use,
        yaxis_title=y_col,
    )
    r_squared = ""
    if show_trend:
        try:
            results = px.get_trendline_results(fig)
            r2 = results.px_fit_results.iloc[0].rsquared
            r_squared = f"R² = {r2:.4f}"
        except Exception:
            pass
    apply_theme(fig, height=340)
    return fig, r_squared


@app.callback(Output("stat-correlation", "figure"), Input("transformation-type", "value"))
def update_correlation(_):
    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[[0, "#f0fdf4"], [0.5, CARETRIA_LIGHT], [1, CARETRIA_TEAL]],
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(title_text="Correlation matrix")
    apply_theme(fig, height=340)
    return fig


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------

server = app.server

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
