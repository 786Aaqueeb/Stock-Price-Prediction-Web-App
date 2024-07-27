# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('MRF.NS', 'TATAPOWER.NS', 'ITC.NS', 'TATASTEEL.NS', 'IRCTC.NS', 'ADANIPOWER.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail(10))

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='red')))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line=dict(color='chartreuse')))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail(10))
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)

# Add actual data to the forecast plot
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="actual_close", line=dict(color='red')))
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)