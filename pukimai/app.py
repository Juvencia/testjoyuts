# app.py - Streamlit Sales Dashboard (with Cohort & Map)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os

st.set_page_config(layout='wide', page_title='Sales Dashboard - UTS (Cohort+Map)')

@st.cache_data
def generate_sample(path='sales.csv', days=180, n_products=30, n_customers=200):
    np.random.seed(42)
    start = datetime.now() - timedelta(days=days)
    products = [f"P{idx:03d}" for idx in range(1, n_products+1)]
    product_names = [f"Product {i}" for i in range(1, n_products+1)]
    categories = ['Electronics','Fashion','Home','Beauty','Sports']
    regions = ['Jakarta','West Java','Central Java','East Java','Bali']
    rows = []
    for i in range(5000):
        order_date = start + timedelta(days=np.random.randint(0, days), hours=np.random.randint(0,24))
        pid = np.random.choice(products)
        pname = product_names[int(pid[1:]) - 1]
        cat = np.random.choice(categories, p=[0.25,0.2,0.2,0.2,0.15])
        price = float(np.round(np.random.lognormal(mean=4.5, sigma=0.7),2))
        qty = int(np.random.choice([1,1,1,2,3], p=[0.6,0.2,0.15,0.03,0.02]))
        customer = f"C{np.random.randint(1, n_customers+1):04d}"
        region = np.random.choice(regions)
        order_id = f"O{np.random.randint(100000,999999)}"
        rows.append([order_date, order_id, pid, pname, cat, price, qty, customer, region])
    df = pd.DataFrame(rows, columns=['order_date','order_id','product_id','product_name','category','price','quantity','customer_id','region'])
    df.to_csv(path, index=False)
    return df

def load_data(path='sales.csv'):
    if not os.path.exists(path):
        df = generate_sample(path)
    else:
        df = pd.read_csv(path, parse_dates=['order_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['revenue'] = df['price'] * df['quantity']
    return df

df = load_data('sales.csv')

# Map of region -> lat/lon for simple plotting
region_coords = {
    'Jakarta': (-6.200000, 106.816666),
    'West Java': (-6.914744, 107.609810),
    'Central Java': (-7.150000, 110.140000),
    'East Java': (-7.250445, 112.768845),
    'Bali': (-8.340539, 115.091950)
}

# Sidebar filters
st.sidebar.header('Filters')
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
date_range = st.sidebar.date_input('Date Range', [min_date, max_date])
cats = st.sidebar.multiselect('Category', options=df['category'].unique(), default=list(df['category'].unique()))
regions = st.sidebar.multiselect('Region', options=df['region'].unique(), default=list(df['region'].unique()))

# apply filters
df = df[(df['order_date'].dt.date >= date_range[0]) & (df['order_date'].dt.date <= date_range[1])]
df = df[df['category'].isin(cats) & df['region'].isin(regions)]

# KPIs
total_revenue = df['revenue'].sum()
total_orders = df['order_id'].nunique()
aov = total_revenue / total_orders if total_orders>0 else 0
growth = 0.0  # placeholder

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"Rp {total_revenue:,.0f}")
col2.metric("Orders", f"{total_orders}")
col3.metric("AOV", f"Rp {aov:,.0f}")
col4.metric("Growth MoM", f"{growth:.1%}")

# Time series revenue
st.subheader("Revenue Over Time")
ts = df.groupby(pd.Grouper(key='order_date', freq='D')).agg({'revenue':'sum','order_id':'nunique'}).reset_index()
fig_ts = px.line(ts, x='order_date', y='revenue', title='Revenue Over Time', markers=True)
st.plotly_chart(fig_ts, use_container_width=True)

# Top products
st.subheader("Top Products by Revenue")
prod = df.groupby(['product_id','product_name']).agg({'revenue':'sum','quantity':'sum'}).reset_index()
prod = prod.sort_values('revenue', ascending=False).head(10)
fig_bar = px.bar(prod, x='revenue', y='product_name', orientation='h', title='Top 10 Products by Revenue', text='revenue')
st.plotly_chart(fig_bar, use_container_width=True)

# Category treemap
st.subheader("Revenue by Category")
cat = df.groupby('category').agg({'revenue':'sum'}).reset_index()
fig_tree = px.treemap(cat, path=['category'], values='revenue', title='Revenue by Category')
st.plotly_chart(fig_tree, use_container_width=True)

# Customer scatter
st.subheader("Customer: Orders vs Revenue")
cust = df.groupby('customer_id').agg({'order_id':'nunique','revenue':'sum'}).reset_index()
fig_scatter = px.scatter(cust, x='order_id', y='revenue', hover_data=['customer_id'], title='Orders vs Revenue per Customer')
st.plotly_chart(fig_scatter, use_container_width=True)

# --- COHORT ANALYSIS ---
st.subheader("Cohort Retention (by Month)")

# Prepare cohort data: cohort_month = month of first purchase, order_month = month of purchase
df_cohort = df.copy()
df_cohort['order_month'] = df_cohort['order_date'].dt.to_period('M').dt.to_timestamp()
first_purchase = df_cohort.groupby('customer_id')['order_date'].min().reset_index().rename(columns={'order_date':'first_order_date'})
first_purchase['cohort_month'] = first_purchase['first_order_date'].dt.to_period('M').dt.to_timestamp()
df_cohort = df_cohort.merge(first_purchase[['customer_id','cohort_month']], on='customer_id', how='left')
df_cohort['period_number'] = ((df_cohort['order_month'].dt.year - df_cohort['cohort_month'].dt.year) * 12 + (df_cohort['order_month'].dt.month - df_cohort['cohort_month'].dt.month))

cohort_data = df_cohort.groupby(['cohort_month','period_number']).agg({'customer_id':'nunique'}).reset_index()
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period_number', values='customer_id')
# convert to retention rate relative to cohort size
cohort_sizes = cohort_pivot.iloc[:,0]
retention = cohort_pivot.divide(cohort_sizes, axis=0).fillna(0).round(3)

# Show heatmap (plotly)
if not retention.empty:
    ret_df = retention.reset_index().melt(id_vars='cohort_month', var_name='period', value_name='retention_rate')
    ret_df['cohort_month_str'] = ret_df['cohort_month'].dt.strftime('%Y-%m')
    fig_cohort = px.density_heatmap(ret_df, x='period', y='cohort_month_str', z='retention_rate', color_continuous_scale='Blues', title='Cohort Retention Heatmap (months since first purchase)')
    st.plotly_chart(fig_cohort, use_container_width=True)
else:
    st.info("Not enough data for cohort analysis. Increase date range or dataset size.")

# --- MAP ---
st.subheader("Sales by Region (Map)")
map_df = df.groupby('region').agg({'revenue':'sum','order_id':'nunique'}).reset_index().rename(columns={'order_id':'orders'})
# add coords
map_df['lat'] = map_df['region'].map(lambda r: region_coords.get(r, (None,None))[0])
map_df['lon'] = map_df['region'].map(lambda r: region_coords.get(r, (None,None))[1])

fig_map = px.scatter_mapbox(map_df, lat='lat', lon='lon', size='revenue', hover_name='region', hover_data=['revenue','orders'], zoom=4, height=400)
fig_map.update_layout(mapbox_style="open-street-map", margin={'r':0,'t':0,'l':0,'b':0})
st.plotly_chart(fig_map, use_container_width=True)

st.markdown(\"\"\"## Insight Box
- Dari cohort retention, perhatikan seberapa cepat pelanggan kembali setelah pembelian pertama — gunakan ini untuk strategi retargeting.
- Peta menunjukkan wilayah dengan revenue tertinggi; fokuskan iklan lokal atau stok di gudang terdekat.
\"\"\")
