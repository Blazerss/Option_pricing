from datetime import datetime as dt
import seaborn
import numpy as np
from streamlit_extras.stylable_container import stylable_container
import streamlit as st
from DX import *
import scipy as sc
from matplotlib import pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide")
st.title("European Option pricing")

st.subheader("Geometric brownian motion and Monte-Carlo simulations")
with st.sidebar:
    st.write(f"**Option parameters**")
    pricing_date = st.text_input(label="Pricing Date (DD/MM/YYYY)")
    if pricing_date:
        pricing_date = dt.strptime(pricing_date,'%d/%m/%Y')

    maturity = st.text_input(label="Maturity (DD/MM/YYYY)")
    if maturity:
        maturity = dt.strptime(maturity, '%d/%m/%Y')

    initial_value = st.number_input(label="Spot price")
    strike = st.number_input(label="Strike")
    short_rate = st.number_input(label="Short Rate")
    volatility = st.number_input(label="Volatility")
    currency = st.text_input("Currency (e.g. EUR,USD,..)")
    currencies = {"EUR": "€","USD": "$" }
    kappa = st.number_input(label="Kappa ")
    theta = st.number_input(label="Theta ")
    sigma = st.number_input(label="initial volatility ")


    st.write("---------------------")
    st.write("**Heatmap parameters**")
    min_spot = st.number_input(label="Min Spot Price")
    max_spot = st.number_input(label="Max Spot Price")

    min_vol = st.number_input(label="Min Volatility")
    max_vol = st.number_input(label="Max Volatility")

if pricing_date and maturity and currency:
    csr = constant_short_rate("crs",short_rate)
    me_gbm = market_environment("me_gbm",pricing_date)
    me_gbm.add_constant("initial_value",initial_value)
    me_gbm.add_constant("final_date",maturity)
    me_gbm.add_constant("volatility",volatility)
    me_gbm.add_constant("currency",currency)
    me_gbm.add_constant("frequency","M")
    me_gbm.add_constant("paths",10000)
    me_gbm.add_curve("discount_curve",csr)
    gbm = geometric_brownian_motion("gbm",mar_env=me_gbm)
    col1,col2= st.columns(2)
    with col1:
        me_call = market_environment("me_call",pricing_date)
        me_call.add_constant("strike",strike)
        me_call.add_constant("currency",currency)
        me_call.add_constant("maturity", maturity)
        payoff_c = 'np.maximum(maturity_value - strike,0)'
        eur_call = valuation_mcs_european("eur_call",gbm,me_call,payoff_func=payoff_c)
        with stylable_container(key="container_with_border_1",
            css_styles="""
                {
                    background-color: green;            
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """,):
            st.write("Call value")
            st.markdown(f"{eur_call.present_value(accuracy=2)} {currencies[currency]}")
    with col2:
        me_put = market_environment("me_put",pricing_date)
        me_put.add_constant("strike",strike)
        me_put.add_constant("currency",currency)
        me_put.add_constant("maturity", maturity)
        payoff_p = 'np.maximum(strike -  maturity_value,0)'
        eur_put = valuation_mcs_european("eur_put",gbm,me_put,payoff_func=payoff_p)
        with stylable_container(key="container_with_border_2",
            css_styles="""
                {
                    
                    background-color: red;            
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """,):
            st.write("Put value")
            st.markdown(f"{eur_put.present_value(accuracy=2)} {currencies[currency]}")



st.subheader("Stochastic Volatility model and Montecarlo simularions")

if pricing_date and maturity and currency :
    csr = constant_short_rate("crs", short_rate)
    me_hes = market_environment("me_hes", pricing_date)
    me_hes.add_constant("initial_value", initial_value)
    me_hes.add_constant("final_date", maturity)
    me_hes.add_constant("currency", currency)
    me_hes.add_constant("frequency", "M")
    me_hes.add_constant("paths", 10000)
    me_hes.add_curve("discount_curve", csr)
    me_hes.add_constant("kappa", kappa)
    me_hes.add_constant("theta", theta)
    me_hes.add_constant('rho' , -0.5)
    me_hes.add_constant("initial_volatility",sigma)
    me_hes.add_constant("volatility_vol",0.2)
    me_hes.add_constant('volatility',None)

    stoc_vol = stochatic_volatility("heston", me_hes)
    col1, col2 = st.columns(2)
    with col1:
        me_call = market_environment("me_call", pricing_date)
        me_call.add_constant("strike", strike)
        me_call.add_constant("currency", currency)
        me_call.add_constant("maturity", maturity)
        payoff_c = 'np.maximum(maturity_value - strike,0)'
        eur_call = valuation_mcs_european("eur_call", stoc_vol, me_call, payoff_func=payoff_c)
        with stylable_container(key="container_with_border_1",
                                css_styles="""
                {
                    background-color: green;            
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """, ):
            st.write("Call value")
            st.markdown(f"{eur_call.present_value(accuracy=2)} {currencies[currency]}")
    with col2:
        me_put = market_environment("me_put", pricing_date)
        me_put.add_constant("strike", strike)
        me_put.add_constant("currency", currency)
        me_put.add_constant("maturity", maturity)
        payoff_p = 'np.maximum(strike -  maturity_value,0)'
        eur_put = valuation_mcs_european("eur_put", stoc_vol, me_put, payoff_func=payoff_p)
        with stylable_container(key="container_with_border_2",
                                css_styles="""
                {

                    background-color: red;            
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """, ):
            st.write("Put value")
            st.markdown(f"{eur_put.present_value(accuracy=2)} {currencies[currency]}")






st.subheader("Heatmap using B&S closed formula")
col1,col2=st.columns(2)

if pricing_date and maturity and currency:

    vols= np.round(np.linspace(min_vol,max_vol, 10), 2)
    prices = np.round( np.linspace(min_spot, max_spot, 10))
    tau = get_year_deltas([pricing_date, maturity])[-1]
    with col1:

        call_value = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                d1 = (np.log(prices[j] / strike) + (short_rate + 0.5 * vols[i] ** 2) * tau) / (
                        vols[i] * (np.sqrt(tau)))
                d2 = d1 - vols[i] * np.sqrt(tau)
                call_value[i, j] = prices[j] * sc.stats.norm.cdf(d1) - strike * np.exp(
                    -short_rate * tau) * sc.stats.norm.cdf(d2)
        x = pd.DataFrame(np.round(call_value, 2), index=vols, columns=prices)

        fig1 = plt.figure(figsize=(8,6))
        plt.title("Call Value")


        ax = seaborn.heatmap(x, annot=True, fmt=".1f", cmap=LinearSegmentedColormap.from_list('rg', ["b", "g"], N=256))
        plt.yticks(rotation=0)
        st.pyplot(fig1)
    with col2:
        put_value = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                d1 = (np.log(prices[j] / strike) + (short_rate + 0.5 * vols[i] ** 2) * tau) / (
                        vols[i] * (np.sqrt(tau)))
                d2 = d1 - vols[i] * np.sqrt(tau)
                put_value[i, j] = - prices[j] * sc.stats.norm.cdf(-d1) + strike * np.exp(
                    -short_rate * tau) * sc.stats.norm.cdf(-d2)
        y = pd.DataFrame(np.round(put_value, 2), index=vols, columns=prices)

        fig2 = plt.figure(figsize=(8,6))
        plt.title("Put Value")



        ax = seaborn.heatmap(y, annot=True, fmt=".1f", cmap=LinearSegmentedColormap.from_list('rg', ["b", "g"], N=256))
        plt.yticks(rotation=0)
        st.pyplot(fig2)





