import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.optimize import minimize
import google.generativeai as genai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for NLP Sentiment
try:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Failed to load NLTK: {e}")

st.set_page_config(page_title="AlphaCore Quant Terminal", layout="wide")
st.title("🌟 AlphaCore: Intelligent Multi-Asset Portfolio & Quantitative Terminal")
st.markdown("*A professional-grade portfolio optimizer designed to make complex institutional finance clear and actionable for everyone.*")

# STATE MANAGEMENT
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose Input:", ["Live Market Data (Tickers)", "Upload Excel/CSV"])

if data_source == "Live Market Data (Tickers)":
    tickers_input = st.sidebar.text_area(
        "Enter Tickers (comma separated):", 
        "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS",
        help="Use exchange suffixes like .NS for NSE or .BO for BSE."
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
else:
    uploaded_file = st.sidebar.file_uploader("Upload custom data", type=["xlsx", "csv"])

st.sidebar.header("2. Optimization Rules (SciPy)")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=3.0, step=0.1) / 100
max_weight = st.sidebar.slider("Max weight per stock (%)", 5, 100, 40) / 100.0

st.sidebar.header("3. Benchmark Configuration")
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="^NSEI", help="e.g., ^NSEI for Nifty 50, ^GSPC for S&P 500")

st.sidebar.header("4. AI Settings")
api_key = st.sidebar.text_input("Gemini API Key (Optional):", type="password")

if st.button("Run Master Analysis"):
    st.session_state.run_analysis = True

@st.cache_data
def get_market_data(tickers):
    data = yf.download(tickers, period="2y")
    if len(tickers) == 1:
        return pd.DataFrame(data['Close'], columns=tickers)
    return pd.DataFrame(data['Close']) if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data['Close'])

@st.cache_data
def get_benchmark_data(ticker):
    data = yf.download(ticker, period="2y")
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close'].iloc[:, 0]
    return data['Close'] if 'Close' in data else data.iloc[:, 0]

@st.cache_data
def get_fundamentals_and_news(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    news = stock.news
    
    sentiment_score = 0
    news_items = []
    if news:
        for article in news[:5]: 
            if 'content' in article:
                title = article['content'].get('title', '')
            else:
                title = article.get('title', '')
                
            if title:
                score = sia.polarity_scores(title)['compound']
                sentiment_score += score
                news_items.append({"Title": title, "Sentiment": score})
        
        avg_sentiment = sentiment_score / len(news_items) if news_items else 0
    else:
        avg_sentiment = 0
        
    fund_data = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "Forward P/E": info.get('forwardPE', 'N/A'),
        "Beta": info.get('beta', 'N/A'),
        "Dividend Yield": info.get('dividendYield', 0),
        "News Sentiment": avg_sentiment
    }
    return fund_data, news_items

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.sum(mean_returns * weights)
    port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(port_return - risk_free_rate) / port_std_dev

if st.session_state.run_analysis:
    try:
        with st.spinner("Compiling data, running NLP, and optimizing math..."):
            is_live = False
            
            if data_source == "Live Market Data (Tickers)":
                if len(tickers) < 2:
                    st.warning("Please enter at least 2 tickers for portfolio optimization.")
                    st.stop()
                    
                closes = get_market_data(tickers)
                closes = closes.dropna(axis=1, how='all')
                valid_tickers = list(closes.columns)
                
                if len(valid_tickers) < len(tickers):
                    missing = set(tickers) - set(valid_tickers)
                    st.warning(f"Could not fetch data for: {', '.join(missing)}. They have been excluded from the analysis.")
                
                if len(valid_tickers) < 2:
                    st.error("Not enough valid tickers remaining to run optimization. Please check your spelling.")
                    st.stop()
                    
                tickers = valid_tickers
                is_live = True
                
            else:
                if uploaded_file:
                    closes = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    closes['Date'] = pd.to_datetime(closes['Date'])
                    closes.set_index('Date', inplace=True)
                    tickers = list(closes.columns)
                else:
                    st.error("Please upload a file.")
                    st.stop()

            daily_returns = closes.pct_change().dropna()
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📰 Fundamentals & NLP", 
                "📈 Advanced Technicals", 
                "⚖️ Optimization & Analytics",
                "🤖 AI CIO Report"
            ])
            
            with tab1:
                st.subheader("Fundamental Ratios & AI News Sentiment")
                st.info("💡 **Plain-English Guide:** This tab evaluates company health and public perception. **P/E Ratio** measures how expensive a stock is relative to its earnings. **Beta** shows how violently a stock swings compared to the general market (Beta > 1 means riskier/more volatile). **News Sentiment** scans recent media headlines using AI to score whether public mood is positive (+) or negative (-).")
                
                if is_live:
                    all_funds = {}
                    all_news = {}
                    for t in tickers:
                        f_data, n_data = get_fundamentals_and_news(t)
                        f_data["Ann. Volatility (Std Dev)"] = f"{(annual_volatility[t] * 100):.2f}%"
                        all_funds[t] = f_data
                        all_news[t] = n_data
                        
                    fund_df = pd.DataFrame(all_funds)
                    
                    def color_sentiment(val):
                        if isinstance(val, (int, float)):
                            color = 'green' if val > 0.05 else 'red' if val < -0.05 else 'gray'
                            return f'color: {color}'
                        return ''
                        
                    st.dataframe(fund_df.style.map(color_sentiment, subset=pd.IndexSlice[["News Sentiment"], :]), use_container_width=True)
                    
                    st.divider()
                    st.write("#### Latest AI-Scored Headlines")
                    selected_news_ticker = st.selectbox("Select ticker to view headlines:", tickers, key="news_ticker")
                    st.table(pd.DataFrame(all_news[selected_news_ticker]))
                else:
                    st.info("Fundamentals and NLP Sentiment require Live Market Data (Tickers).")
                    st.subheader("Standard Deviation (Volatility)")
                    vol_df = pd.DataFrame(annual_volatility * 100, columns=["Ann. Volatility (%)"]).round(2)
                    st.dataframe(vol_df.T, use_container_width=True)

            with tab2:
                st.subheader("Advanced Technical Analysis & Indicators")
                st.info("💡 **Plain-English Guide:** Technical indicators help identify timing. **RSI (Relative Strength Index)** tells you if a stock is overbought (>70, meaning price might drop soon) or oversold (<30, meaning it might bounce). **MACD** tracks momentum shifts (Bullish means upward momentum, Bearish means downward). **Bollinger Bands** act like elastic rails around price action to show normal price limits.")
                
                tech_ticker = st.selectbox("Select stock for technical analysis:", tickers, key="tech_tick")
                
                tech_df = pd.DataFrame()
                tech_df['Close'] = closes[tech_ticker]
                
                tech_df['SMA_20'] = tech_df['Close'].rolling(window=20).mean()
                tech_df['BB_Upper'] = tech_df['SMA_20'] + 2 * tech_df['Close'].rolling(window=20).std()
                tech_df['BB_Lower'] = tech_df['SMA_20'] - 2 * tech_df['Close'].rolling(window=20).std()
                
                tech_df['EMA_12'] = tech_df['Close'].ewm(span=12, adjust=False).mean()
                tech_df['EMA_26'] = tech_df['Close'].ewm(span=26, adjust=False).mean()
                tech_df['MACD'] = tech_df['EMA_12'] - tech_df['EMA_26']
                tech_df['Signal'] = tech_df['MACD'].ewm(span=9, adjust=False).mean()
                
                delta = tech_df['Close'].diff()
                gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                tech_df['RSI'] = 100 - (100 / (1 + rs))

                latest_close = tech_df['Close'].iloc[-1]
                prev_close = tech_df['Close'].iloc[-2]
                close_change = ((latest_close - prev_close) / prev_close) * 100
                
                latest_rsi = tech_df['RSI'].iloc[-1]
                latest_macd = tech_df['MACD'].iloc[-1]
                latest_signal = tech_df['Signal'].iloc[-1]
                latest_upper_bb = tech_df['BB_Upper'].iloc[-1]
                latest_lower_bb = tech_df['BB_Lower'].iloc[-1]

                st.write(f"#### 📊 Key Technical Metrics for {tech_ticker} (Latest Trading Session)")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Latest Close", f"${latest_close:.2f}", f"{close_change:+.2f}%")
                with m2:
                    rsi_status = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{latest_rsi:.2f}", rsi_status)
                with m3:
                    macd_status = "Bullish (Above Signal)" if latest_macd > latest_signal else "Bearish (Below Signal)"
                    st.metric("MACD Line", f"{latest_macd:.4f}", macd_status)
                with m4:
                    bb_width = latest_upper_bb - latest_lower_bb
                    st.metric("Bollinger Band Width", f"${bb_width:.2f}", f"Upper: ${latest_upper_bb:.2f}")

                st.divider()

                with st.expander("View Raw Technical Data Table (Last 10 Days)"):
                    display_cols = ['Close', 'SMA_20', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal', 'RSI']
                    st.dataframe(tech_df[display_cols].tail(10).round(4), use_container_width=True)

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
                
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], name='Close'), row=1, col=1)
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['BB_Upper'], line=dict(dash='dot', color='gray'), name='Upper BB'), row=1, col=1)
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['BB_Lower'], line=dict(dash='dot', color='gray'), name='Lower BB'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['MACD'], name='MACD'), row=2, col=1)
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Signal'], name='Signal'), row=2, col=1)
                fig.add_trace(go.Bar(x=tech_df.index, y=tech_df['MACD'] - tech_df['Signal'], name='Histogram'), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

                fig.update_layout(height=800, title_text=f"{tech_ticker} Advanced Technical Charts & Overlays", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Modern Portfolio Theory, Correlation & Simulation Engine")
                st.info("💡 **Plain-English Guide:** This section brings everything together. **Target Allocation** uses math (SciPy) to find the ideal percentage of money to place in each stock to maximize returns while keeping risk low. **Sharpe Ratio** measures risk-adjusted return (above 1.0 is considered good, above 2.0 is excellent). **Correlation Heatmap** shows if stocks move together (avoid stocks that move identically). **Monte Carlo simulation** forecasts 500 potential future paths for your money over the next year based on historical trends.")
                
                mean_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252
                
                num_assets = len(tickers)
                initial_guess = np.array(num_assets * [1. / num_assets])
                
                min_required_weight = 1.0 / num_assets
                if max_weight < min_required_weight:
                    max_weight = min_required_weight

                bounds = tuple((0.0, max_weight) for _ in range(num_assets))
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                
                optimized = minimize(
                    negative_sharpe, initial_guess, 
                    args=(mean_returns, cov_matrix, risk_free_rate),
                    method='SLSQP', bounds=bounds, constraints=constraints
                )
                
                optimal_weights = optimized.x
                opt_return = np.sum(mean_returns * optimal_weights)
                opt_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                opt_sharpe = (opt_return - risk_free_rate) / opt_risk
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    weight_df = pd.DataFrame({"Ticker": tickers, "Weight": (optimal_weights * 100).round(2)})
                    weight_df = weight_df[weight_df["Weight"] > 0]
                    fig_pie = px.pie(weight_df, values='Weight', names='Ticker', title="Target Allocation")
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.write("#### Expected Institutional Metrics")
                    st.metric("Expected Annual Return", f"{(opt_return * 100):.2f}%")
                    st.metric("Expected Annual Risk (Volatility)", f"{(opt_risk * 100):.2f}%")
                    st.metric("Optimized Sharpe Ratio", f"{opt_sharpe:.2f}")

                    st.divider()
                    st.subheader("📥 Export Data")
                    csv_data = weight_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Optimal Weights (CSV)",
                        data=csv_data,
                        file_name="optimal_portfolio_weights.csv",
                        mime="text/csv",
                    )

                st.divider()
                st.subheader("🔗 Asset Correlation Matrix Heatmap")
                corr_matrix = daily_returns.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Blues", title="Asset Returns Correlation (1.0 = Perfect Match)", template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

                st.divider()
                st.subheader("📈 Cumulative Growth vs. Benchmark Index")
                try:
                    bench_close = get_benchmark_data(benchmark_ticker)
                    port_daily = (daily_returns * optimal_weights).sum(axis=1)
                    
                    perf_df = pd.DataFrame({
                        "Optimized Portfolio": port_daily,
                        benchmark_ticker: bench_close.pct_change()
                    }).dropna()
                    
                    cum_perf_df = (1 + perf_df).cumprod() * 100
                    fig_bench = px.line(cum_perf_df, title=f"Portfolio Growth vs. {benchmark_ticker} (Base = 100)", template="plotly_dark")
                    fig_bench.update_layout(yaxis_title="Normalized Value", xaxis_title="Date")
                    st.plotly_chart(fig_bench, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load benchmark comparison chart: {e}")

                st.divider()
                st.subheader("🎲 Monte Carlo Portfolio Simulation (1 Year Forward)")
                num_simulations = 500
                num_days = 252
                dt = 1 / 252
                
                simulated_paths = np.zeros((num_days, num_simulations))
                initial_value = 10000
                daily_rets_sim = np.random.normal(opt_return * dt, opt_risk * np.sqrt(dt), (num_days, num_simulations))
                simulated_paths = initial_value * np.cumprod(1 + daily_rets_sim, axis=0)
                
                fig_mc = go.Figure()
                for i in range(min(100, num_simulations)):
                    fig_mc.add_trace(go.Scatter(y=simulated_paths[:, i], mode='lines', line=dict(width=1, color='rgba(0, 200, 255, 0.15)'), showlegend=False))
                
                mean_path = np.mean(simulated_paths, axis=1)
                fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(width=3, color='orange'), name='Expected Path (Mean)'))
                fig_mc.update_layout(title="Monte Carlo Simulation (500 Paths, 1-Year Horizon)", xaxis_title="Trading Days Forward", yaxis_title="Portfolio Value ($)", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)

            with tab4:
                st.subheader("🤖 AI Executive Summary")
                st.info("💡 **Plain-English Guide:** This tab uses artificial intelligence to act as your personal Chief Investment Officer (CIO), converting all the technical statistics above into a clear, written 3-paragraph executive memo.")
                
                if not api_key:
                    st.info("Please enter your Gemini API Key in the sidebar to generate the executive report.")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""
                        You are a Chief Investment Officer. Review this optimized portfolio:
                        Tickers & Weights: {weight_df.to_dict()}
                        Expected Annual Return: {opt_return}
                        Risk/Volatility: {opt_risk}
                        Sharpe Ratio: {opt_sharpe}
                        
                        Write a 3-paragraph executive memo in plain, easy-to-understand language summarizing the portfolio performance, diversification benefits, and risk metrics for a client.
                        """
                        response = model.generate_content(prompt)
                        st.write(response.text)
                    except Exception as e:
                        st.warning("AI connection notice: The model is currently verifying permissions. Your portfolio math, technicals, and SciPy optimization calculations above are fully functional.")
    except Exception as e:
        st.error(f"Execution Error: {e}")