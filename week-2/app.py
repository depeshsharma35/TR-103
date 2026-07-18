# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="Sales Forecast Dashboard", page_icon="📈", layout="wide")

# App title
st.title("📈 Sales Forecast Dashboard")
st.markdown("Explore the sales data, forecasting model predictions, and feature importance.")

# Function to load data with caching
@st.cache_data
def load_data():
    # Attempt to load relevant datasets
    data_dir = "."
    forecast_path = os.path.join(data_dir, "forecast_results.csv")
    feature_importance_path = os.path.join(data_dir, "feature_importance.csv")
    
    forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else None
    importance_df = pd.read_csv(feature_importance_path) if os.path.exists(feature_importance_path) else None
    
    return forecast_df, importance_df

# Load data
forecast_df, importance_df = load_data()

if forecast_df is None:
    st.error("No forecast data found. Make sure 'forecast_results.csv' is generated.")
else:
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview & Forecast", "Feature Importance"])

    # Convert Date column to datetime if it exists
    if 'Date' in forecast_df.columns:
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        forecast_df = forecast_df.sort_values('Date')

    if page == "Overview & Forecast":
        st.header("Actual vs Predicted Sales")
        
        # Calculate high-level metrics
        actual_total = forecast_df['Actual'].sum()
        predicted_total = forecast_df['Predicted'].sum()
        mae = (forecast_df['Actual'] - forecast_df['Predicted']).abs().mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Actual Sales", f"${actual_total:,.2f}")
        col2.metric("Total Predicted Sales", f"${predicted_total:,.2f}")
        col3.metric("Mean Absolute Error", f"${mae:,.2f}")
        
        st.divider()

        # Plotly Line Chart for Forecast Comparison
        if 'Date' in forecast_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Actual'], mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted'], mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
            
            fig.update_layout(
                title='Actual vs Predicted Sales Over Time',
                xaxis_title='Date',
                yaxis_title='Sales',
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show Raw Data
            with st.expander("View Raw Forecast Data"):
                st.dataframe(forecast_df)
        else:
            st.warning("No 'Date' column found in forecast_results.csv. Displaying basic line chart.")
            st.line_chart(forecast_df[['Actual', 'Predicted']])

    elif page == "Feature Importance":
        st.header("Model Feature Importance")
        st.markdown("Discover which features are driving the sales predictions.")
        
        if importance_df is None:
            st.error("No feature importance data found. Make sure 'feature_importance.csv' is generated.")
        else:
            # Plotly Bar Chart
            fig_importance = px.bar(
                importance_df.head(15), 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title='Top Feature Importances',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
            st.plotly_chart(fig_importance, use_container_width=True)
            
            with st.expander("View Feature Importance Data"):
                st.dataframe(importance_df)
