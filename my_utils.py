# python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 02:15:50 2019

@author: yz3380@columbia.edu
"""

import os
import glob
import operator
import itertools
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as so
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pandas_datareader import data as web
from pandas_datareader._utils import RemoteDataError
from matplotlib.lines import Line2D

# check inputs
def Check_input(Time_start, Time_end, VaR_p, ES_p, Initial_value, Window, Horizon_period, Parameter):
    assert(isinstance(Time_start, pd.Timestamp))
    assert(isinstance(Time_end, pd.Timestamp))
    assert(Time_start < Time_end)
    assert(isinstance(VaR_p, (int, float)))
    assert(isinstance(ES_p, (int, float)))
    assert(0 < VaR_p < 1)
    assert(0 < ES_p < 1)
    assert(isinstance(Initial_value, (int, float)))
    assert(Initial_value > 0)
    assert(isinstance(Window, (int, float)))
    assert(Window > 0)
    assert(isinstance(Horizon_period, int))
    assert(Horizon_period > 0)
    assert(Parameter in ('Window', 'Exponential'))
    return

# read underlyings and options data from current working directory
# can read unlimited number of files, as long as correct CSV format is provided
def Read_option(path):
    try:
        all_files = glob.glob(os.path.join(path, 'Option_Data', '*.csv'))
        
        option_df = {}
        for file in all_files:
            stock_name = pd.read_csv(file).columns[0].upper() + '_option'
            df = pd.read_csv(file, skiprows=1)
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
            df = df.fillna(method='ffill')
            df.iloc[:, 1:] = df.iloc[:, 1:] / 100
            option_df[stock_name] = df
    
    except FileNotFoundError:
        raise FileNotFoundError('Invalid directory or file name!')
    else:
        return option_df

# read portfolio positions
def Read_position(path, option_df):
    try:
        portfolio = os.path.join(path, 'Portfolio.txt')
        stocks = {}
        options = {}
        with open(portfolio) as f:
            for line in f.readlines():
                if line[0] in ['#', '-', '\n']:
                    continue
                else:
                    line = line.strip()
                    line = line.replace(' ', '')
                    content = line.split(',')
                    if len(content) == 2:                                        # read stock
                        assert(isinstance(content[0], str))
                        assert(isinstance(float(content[1]), float))
                        stocks[content[0].upper()] = float(content[1])
                    elif len(content) == 4:                                      # read option
                        assert(isinstance(content[0], str))
                        assert(isinstance(float(content[1]), float))
                        assert(content[2].lower() in ('call', 'put'))
                        assert(int(content[3]) in (3, 6, 12))
                        name = content[0].upper() + '_option'
                        options[name] = (float(content[1]), content[2].lower(), int(content[3]))
                        if not name in option_df.keys():
                            raise NameError
                    else:
                        raise ValueError
            assert(len(stocks) + len(options) > 0)        
    
    except (AssertionError, ValueError):
        raise ValueError('Invalid inputs! Please check your Portfolio.txt')
    except FileNotFoundError:
        raise FileNotFoundError('Input files not exist!')
    except NameError:
        raise NameError('Options must have provided underlying stocks!')
    else:
        return stocks, options
    
# adjust variables according to portfolio positions
def Adjust_option_time(option_df, options, start, end):
    opt_start = option_df[next(iter(option_df))].index[0]
    opt_end = option_df[next(iter(option_df))].index[-1]
    
    if bool(options):                                                            # portfolio include options, adjust time period
        if start < opt_start:
            start = opt_start
        if end > opt_end:
            end = opt_end
    
    return start, end

# download stocks price from Yahoo
def Read_stock(stocks, start, end):
    stock_data = {}
    adj_start = start
    adj_end = end
    if adj_start < pd.Timestamp('1980-01-01'):                                   # set lower bound for starting period
        adj_start = pd.Timestamp('1980-01-01')
    try:
        for ticker in stocks.keys():
            df = pd.DataFrame()
            df[ticker] = web.DataReader(ticker, 'yahoo', adj_start, adj_end)['Close']
            df = df.fillna(method='ffill')
            stock_data[ticker] = df
    
            if adj_start < stock_data[ticker].index[0]:
                adj_start = stock_data[ticker].index[0]
            if adj_end > stock_data[ticker].index[-1]:
                adj_end = stock_data[ticker].index[-1]
                
        for ticker in stock_data.keys():                                         # adjust dataframe to same length
            stock_data[ticker] = stock_data[ticker].truncate(before=adj_start, after=adj_end)
        
        if len(stock_data) > 0:
            combine_stock = pd.concat([stock_data[x] for x in stock_data.keys()], axis = 1)
            combine_stock = combine_stock.fillna(method='ffill')
            for ticker in combine_stock.columns:
                stock_data[ticker] = combine_stock[ticker]
    
    except OverflowError as err:
        raise OverflowError(err)
    except RemoteDataError as err:
        raise RemoteDataError(err)
    except KeyError as err:
        raise KeyError(err)
    else:
        return stock_data, adj_start, adj_end
    
# load 1 year Treasury bill rate
def Read_interest(path, start, end):
    try:
        FRB = os.path.join(path, 'Option_Data', 'FRB_H15.csv')
 
        df = pd.read_csv(FRB)
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1]
        df = df.fillna(method='ffill')
        df = df.truncate(before=start, after=end)
        df= df / 100
    
    except FileNotFoundError:
        raise FileNotFoundError('(Interest rate)Invalid directory or file name!')
    else:
        return df

# Construct portfolio value and 1 day log return data, also truncated option data frame
def Portfolio_return(stock_df, option_df, int_rate, stocks, options, start, end):
    for ticker in option_df.keys():
        option_df[ticker] = option_df[ticker].truncate(before=start, after=end)  # truncate option data to adjusted time period
    
    if len(stock_df) > 0:
        index = stock_df[next(iter(stock_df))].index
    else:
        index = option_df[next(iter(option_df))].index
    N = len(index)
    
    portfolio = np.zeros(N)
    for ticker in stocks:
        portfolio += stock_df[ticker].values * stocks[ticker] #
    for ticker in options:
        iv_index = 3 if options[ticker][1] == 'call' else 0                      # set index for option implied volatility
        iv_lib = dict(zip([3, 6, 12], [1, 2, 3]))
        iv_index = iv_index + iv_lib[options[ticker][2]]
        portfolio += Black_Scholes(option_df[ticker].iloc[:, 0].values, option_df[ticker].iloc[:, 0].values, int_rate.values,\
                                   option_df[ticker].iloc[:,iv_index].values, options[ticker][2]/12, options[ticker][1])\
                    * options[ticker][0]
    
    if portfolio[0] > 0:                                                         # set long / short state
        long_portfolio = True
        portfolio = portfolio.clip(min=0.01)                                     # avoid negative value in long portfolio
    else:
        long_portfolio = False
        portfolio = portfolio.clip(max=-0.01)
    
    portfolio_past = portfolio[:-1]                                              # calculate one day log return
    portfolio = portfolio[1:]
    log_return =np.log(portfolio / portfolio_past)
    p_return = pd.DataFrame({'Price':portfolio})
    p_return['Return'] = log_return
    p_return.set_index(index[1:], inplace = True)
    p_return.fillna(method='ffill')
    
    for ticker in stock_df.keys():                                               # update the stock data
        price_past = stock_df[ticker].values[:-1]                                         
        price = stock_df[ticker].values[1:]
        log_return = np.log(price / price_past)
        s_return = pd.DataFrame({'Price':price})
        s_return['Return'] = log_return
        s_return.set_index(index[1:], inplace = True)
        stock_df[ticker] = s_return
    for ticker in options.keys():                                                # add option underlying to stock data
        name = ticker.replace('_option', '')
        if not name in stock_df.keys():
            price_past = option_df[ticker].iloc[:, 0].values[:-1]                                         
            price = option_df[ticker].iloc[:, 0].values[1:]
            log_return = np.log(price / price_past)
            s_return = pd.DataFrame({'Price':price})
            s_return['Return'] = log_return
            s_return.set_index(index[1:], inplace = True)
            stock_df[name] = s_return
        
    return p_return, option_df, stock_df, long_portfolio

# construct relative change data frame
def Historical_change(port_df, period):
    index = port_df.index
    portfolio = port_df.iloc[:, 0].values
    
    if port_df.shape[0] < period:
        raise ValueError('Horizon length cannot be longer than total data!')
    
    portfolio_past = portfolio[:-period]                                         # calculate k day difference
    portfolio = portfolio[period:]
    rel_return = portfolio / portfolio_past
    abs_return = portfolio - portfolio_past
    
    p_return = pd.DataFrame({'Price':portfolio})
    p_return['Relative'] = rel_return
    p_return['Absolute'] = abs_return
    p_return.set_index(index[period:], inplace = True)
    
    return p_return

# calibrate GBM parameter, window version
def Calibrate(df, l, period):
    T = round(l * 252)
    N = df.shape[0]
    if N <= T + period:
        raise ValueError('Window size cannot be longer than total data!')
    
    drift = []
    sigma = []
    for start in range(N - T + 1):
        end = start + T
        mu = sum(df.iloc[start:end, 1]) / T
        var = sum(df.iloc[start:end, 1] ** 2) / T - mu ** 2
        sigma.append(np.sqrt(var) * np.sqrt(252))
        drift.append(mu * 252 + sigma[-1] ** 2 / 2)
    
    data = {'drift':drift, 'sigma':sigma}
    parameter = pd.DataFrame(data, index = df.index[T - 1:])
    return parameter

# discrepancy function in estimating lambda
def Estimate_lambda(_lambda, l):    
    N = 252 * l
    Distance = (2 *_lambda ** (N + 1) + 2 * _lambda ** N - _lambda * (N + 1) + N - 1) / ((_lambda + 1) * N)
    return Distance

# calibrate GBM parameter, equivalent exponential version
def Exp_Calibrate(df, l, period):
    T = round(l * 252)
    N = df.shape[0]
    if N <= T + period:
        raise ValueError('Window size cannot be longer than total data!')
    
    optimize = so.minimize(Estimate_lambda, x0=1, args=l)                        # calculate lambda
    _lambda = float(optimize.x)
    drift = []
    sigma = []
    
    for start in range(N - T + 1):
        end = start + T
        weight = np.array(list(itertools.accumulate([_lambda] * (end), 
                                                    operator.mul))) * (1 - _lambda) / _lambda
        weight = np.fliplr([weight])[0]
        #weight = [_lambda ** (end - i - 1) * (1 - _lambda) for i in range(end)] # loop version, slow in computation
        mu = sum(df.iloc[:end, 1] * weight)
        var = sum(df.iloc[:end, 1] ** 2 * weight) - mu ** 2
        sigma.append(np.sqrt(var) * np.sqrt(252))
        drift.append(mu * 252 + sigma[-1] ** 2 / 2)
    
    data = {'drift':drift, 'sigma':sigma}
    result = pd.DataFrame(data, index = df.index[T - 1:])
    return result

# calibrate window GBM parameters for all stocks
def Calibrate_all(stock_df, l, period):
    gbm_stocks = {}
    for ticker in stock_df.keys():
        gbm_stocks[ticker] = Calibrate(stock_df[ticker], l, period)
    return gbm_stocks

# calibrate exp GBM parameters for all stocks
def Calibrate_all_EW(stock_df, l, period):
    gbm_stocks_ew = {}
    for ticker in stock_df.keys():
        gbm_stocks_ew[ticker] = Exp_Calibrate(stock_df[ticker], l, period)
    return gbm_stocks_ew

# calculate correlation
def Calculate_rho(stock_df, l):
    T = round(l * 252)
    N = stock_df[next(iter(stock_df))].shape[0]
    combine_df = pd.DataFrame()
    for ticker in stock_df.keys():
        combine_df[ticker] = stock_df[ticker]['Return']
    
    rho = []
    for start in range(N - T + 1):
        end = start + T
        df = combine_df.iloc[start:end, :]
        rho.append(df.corr())
    
    return rho

# Black Scholes option pricing, array version
def Black_Scholes(S_0, K, r, sigma, T, opt_type):                                # S_0, K, r, sigma are arrays
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    c_price = st.norm.cdf(d1) * S_0 - K * np.exp(-r * T) * st.norm.cdf(d2)
    p_price = K * np.exp(-r * T) * st.norm.cdf(-d2) - st.norm.cdf(-d1) * S_0
    
    if opt_type == 'call':
        return c_price
    else:
        return p_price
    
# long and short VaR, array version
def Long_VaR(V, mu, sigma, period, level):                                       # mu, sigma array
    VaR = V - V * np.exp((mu - 0.5 * sigma ** 2) * period + sigma * np.sqrt(period) * st.norm.ppf(1 - level))
    return VaR

def Short_VaR(V, mu, sigma, period, level):
    VaR = V * np.exp((mu - 0.5 * sigma ** 2) * period + sigma * np.sqrt(period) * st.norm.ppf(level)) - V
    return VaR

# parametric GBM VaR
def Parametric_VaR_GBM(parameters, V, period, level, state):
    period = period / 252
    index = parameters.index
    mu = parameters.iloc[:, 0].values
    sigma = parameters.iloc[:, 1].values
    
    if state == True:                                                            # long portfolio
        VaR = Long_VaR(V, mu, sigma, period, level)
    else:                                                                        # short portfolio
        VaR = Short_VaR(V, mu, sigma, period, level)
    
    result = pd.DataFrame({'GBM VaR':VaR}, index = index)
    return result

# parametric VaR, approximate Normal, STOCKS ONLY
def Parametric_VaR_Normal(stock_df, stock_parameters, stocks, rho, V, l, period, level, state):
    period = period / 252
    index = stock_parameters[next(iter(stock_parameters))].index
    N = len(index)
    T = round(l * 252)
    portfolio_value = np.sum([stock_df[x].iloc[T - 1:, 0].values for x in stocks.keys()])
    
    ev = np.zeros(N)
    ev2 = np.zeros(N)
    portfolio_value = np.zeros(N)
    for ticker1 in stocks.keys():                                                # calculate normal approximate parameters
        portfolio_value += stock_df[ticker1].iloc[T - 1:, 0].values * stocks[ticker1]
        a = stocks[ticker1] * stock_df[ticker1].iloc[T - 1:, 0].values
        mu1 = stock_parameters[ticker1].iloc[:, 0].values * period
        sigma1 = stock_parameters[ticker1].iloc[:, 1].values * np.sqrt(period)
        ev += a * np.exp(mu1)
        ev2 += a ** 2 * np.exp(2 * mu1 + sigma1 ** 2)
        for ticker2 in stocks.keys():
            if ticker2 != ticker1:
                b = stocks[ticker2] * stock_df[ticker2].iloc[T - 1:, 0].values
                mu2 = stock_parameters[ticker2].iloc[:, 0].values * period
                sigma2 = stock_parameters[ticker2].iloc[:, 1].values * np.sqrt(period)
                
                rho_ab = np.array([x[ticker1][ticker2] for x in rho])            # get correlation array
                ev2 += a * b * np.exp(mu1 + mu2 + rho_ab * sigma1 * sigma2)
    
    ev2 = ev2 - ev ** 2
    sd = np.sqrt(ev2)
    
    if state == True:                                                            # long portfolio
        VaR = V - (ev + sd * st.norm.ppf(1 - level)) * V / portfolio_value
    else:                                                                        # short portfolio
        VaR = - (ev + sd * st.norm.ppf(level)) * V / portfolio_value + V
    result = pd.DataFrame({'Normal VaR':VaR}, index = index)
    return result

# Calculate historical relative and absolute VaR
def Historical_VaR(his_df, V, l, level, state):
    T = round(l * 252)
    N = his_df.shape[0]
    rel_VaR = []
    abs_VaR = []
    for start in range(N - T + 1):
        end = start + T
        if state == True:
            relative = np.quantile(his_df.iloc[start:end, 1], 1 - level, interpolation='lower')
            absolute = np.quantile(his_df.iloc[start:end, 2], 1 - level, interpolation='lower')  
            rel_VaR.append(V - V * relative)
            abs_VaR.append(-absolute * V / his_df.iloc[end - 1, 0])
        else:
            relative = np.quantile(his_df.iloc[start:end, 1], level, interpolation='higher')
            absolute = np.quantile(his_df.iloc[start:end, 2], level, interpolation='higher')  
            rel_VaR.append(V * relative - V)
            abs_VaR.append(-absolute * V / his_df.iloc[end - 1, 0])
    
    result = pd.DataFrame({'Historical rel VaR':rel_VaR, 'Historical abs VaR':abs_VaR}, index = his_df.index[T - 1:])
    return result

# calculate MC VaR as one portfolio
def MC_VaR_One(p_parameters, V, period, level, sample, state):
    period = period / 252
    N = p_parameters.shape[0]

    VaR = []
    for i in range(N):
        mu = p_parameters.iloc[i, 0] * period
        sigma = p_parameters.iloc[i, 1] * np.sqrt(period)
        r = np.random.normal(0, sigma, sample)
        if state == True:
            log_r = np.quantile(r, 1 - level)
            VaR.append(V - V * np.exp(mu - 0.5 * sigma ** 2 + log_r))
        else:
            log_r = np.quantile(r, level)
            VaR.append(V * np.exp(mu - 0.5 * sigma ** 2 + log_r) - V)
    
    result = pd.DataFrame({'MC one VaR':VaR}, index = p_parameters.index)
    return result

# calculate MC VaR by simulating multiple stocks
def MC_VaR_Multi(s_parameters, port_df, option_df, stock_df, stocks, options, 
                 rho, int_rate, V, l, period, level, sample, state):
    period = period / 252
    T = round(l * 252)
    index = s_parameters[next(iter(s_parameters))].index
    N = len(index)
    
    VaR = [] 
    for i in range(N):
        mu = np.array([0 for x in stock_df.keys()])
        sigma = np.array([s_parameters[x].iloc[i, 1] for x in stock_df.keys()])                                         
        cov = (np.diag(sigma) @ np.diag(sigma).T) @ rho[i].values
        
        r = pd.DataFrame(np.random.multivariate_normal(mu, cov, sample, tol=1))  # stock price using multi variate normal
        r.columns = [ticker for ticker in stock_df.keys()]
        
        portfolio = np.zeros(sample)
        for ticker in stocks:
            price = np.exp((s_parameters[ticker].iloc[i, 0] - 0.5 * s_parameters[ticker].iloc[i, 1] ** 2) * period
                           + np.sqrt(period) * r[ticker].values) * stock_df[ticker].iloc[T-1+i, 0]
            portfolio += price * stocks[ticker]
        for ticker in options.keys():
            name = ticker.replace('_option', '')
            iv_index = 3 if options[ticker][1] == 'call' else 0                  # set index for option implied volatility
            iv_lib = dict(zip([3, 6, 12], [1, 2, 3]))
            iv_index = iv_index + iv_lib[options[ticker][2]]
            
            price = np.exp((s_parameters[name].iloc[i, 0] - 0.5 * s_parameters[name].iloc[i, 1] ** 2) * period
                           + np.sqrt(period) * r[name].values) * stock_df[name].iloc[T-1+i, 0]
            portfolio += Black_Scholes(price, price, int_rate.values[T-1+i], option_df[ticker].iloc[T-1+i,iv_index], \
                                       options[ticker][2]/12, options[ticker][1]) * options[ticker][0]
        
        portfolio = portfolio * V / port_df.iloc[T-1+i, 0]
        
        if state == True:
            VaR_temp = np.quantile(portfolio, 1 - level, interpolation = 'lower')
            VaR.append(V - VaR_temp)
        else:
            VaR_temp = np.quantile(portfolio, level, interpolation = 'higher')
            VaR.append(VaR_temp - V)
        
    result = pd.DataFrame({'MC multi VaR':VaR}, index = index)
    return result

# Calculate historical relative and absolute ES
def Historical_ES(his_df, V, l, level, state):
    T = round(l * 252)
    N = his_df.shape[0]
    rel_ES = []
    abs_ES = []
    for start in range(N - T + 1):
        end = start + T
        if state == True:
            relative = np.quantile(his_df.iloc[start:end, 1], 1 - level, interpolation='lower')
            absolute = np.quantile(his_df.iloc[start:end, 2], 1 - level, interpolation='lower')  
            
            value1 = V * his_df.iloc[start:end, 1].values
            value1 = value1[value1 < relative * V]
            rel_ES.append(V - np.mean(value1))
            
            value2 = -his_df.iloc[start:end, 2].values * V / his_df.iloc[end - 1, 0]
            value2 = value2[value2 > -absolute * V / his_df.iloc[end - 1, 0]]
            abs_ES.append(np.mean(value2))
        else:
            relative = np.quantile(his_df.iloc[start:end, 1], level, interpolation='higher')
            absolute = np.quantile(his_df.iloc[start:end, 2], level, interpolation='higher')  
            
            value1 = V * his_df.iloc[start:end, 1].values
            value1 = value1[value1 > relative * V]
            rel_ES.append(np.mean(value1) - V)
            
            value2 = - his_df.iloc[start:end, 2].values * V / his_df.iloc[end - 1, 0]
            value2 = value2[value2 > - absolute * V / his_df.iloc[end - 1, 0]]
            abs_ES.append(np.mean(value2))
    
    result = pd.DataFrame({'Historical rel ES':rel_ES, 'Historical abs ES':abs_ES}, 
                          index = his_df.index[T - 1:])
    return result

# calculate MC ES as one portfolio
def MC_ES_One(p_parameters, V, period, level, sample, state):
    period = period / 252
    N = p_parameters.shape[0]

    ES = []
    for i in range(N):
        mu = p_parameters.iloc[i, 0] * period
        sigma = p_parameters.iloc[i, 1] * np.sqrt(period)
        r = np.random.normal(0, sigma, sample)
        if state == True:
            log_r = np.quantile(r, 1 - level)
            VaR = V - V * np.exp(mu - 0.5 * sigma ** 2 + log_r)
            value1 = V - V * np.exp(mu - 0.5 * sigma ** 2 + r)
            value1 = value1[value1 > VaR]
            ES.append(np.mean(value1))
        else:
            log_r = np.quantile(r, level)
            VaR = V * np.exp(mu - 0.5 * sigma ** 2 + log_r) - V
            value2 = V * np.exp(mu - 0.5 * sigma ** 2 + r) - V
            value2 = value2[value2 > VaR]
            ES.append(np.mean(value2))
    
    result = pd.DataFrame({'MC one ES':ES}, index = p_parameters.index)
    return result

# calculate MC ES by simulating multiple stocks
def MC_ES_Multi(s_parameters, port_df, option_df, stock_df, stocks, options, 
                rho, int_rate, V, l, period, level, sample, state):
    period = period / 252
    T = round(l * 252)
    index = s_parameters[next(iter(s_parameters))].index
    N = len(index)
    
    ES = [] 
    for i in range(N):
        mu = np.array([0 for x in stock_df.keys()])
        sigma = np.array([s_parameters[x].iloc[i, 1] for x in stock_df.keys()])                                         
        cov = (np.diag(sigma) @ np.diag(sigma).T) @ rho[i].values
        
        r = pd.DataFrame(np.random.multivariate_normal(mu, cov, sample, tol=1))  # stock price using multi variate normal
        r.columns = [ticker for ticker in stock_df.keys()]
        
        portfolio = np.zeros(sample)
        for ticker in stocks:
            price = np.exp((s_parameters[ticker].iloc[i, 0] - 0.5 * s_parameters[ticker].iloc[i, 1] ** 2) * period
                           + np.sqrt(period) * r[ticker].values) * stock_df[ticker].iloc[T-1+i, 0]
            portfolio += price * stocks[ticker]
        for ticker in options.keys():
            name = ticker.replace('_option', '')
            iv_index = 3 if options[ticker][1] == 'call' else 0                  # set index for option implied volatility
            iv_lib = dict(zip([3, 6, 12], [1, 2, 3]))
            iv_index = iv_index + iv_lib[options[ticker][2]]
            
            price = np.exp((s_parameters[name].iloc[i, 0] - 0.5 * s_parameters[name].iloc[i, 1] ** 2) * period
                           + np.sqrt(period) * r[name].values) * stock_df[name].iloc[T-1+i, 0]
            portfolio += Black_Scholes(price, price, int_rate.values[T-1+i], option_df[ticker].iloc[T-1+i,iv_index], \
                                       options[ticker][2]/12, options[ticker][1]) * options[ticker][0]
        
        portfolio = portfolio * V / port_df.iloc[T-1+i, 0]
        
        if state == True:
            VaR = V - np.quantile(portfolio, 1 - level, interpolation = 'lower')
            value1 = V - portfolio
            value1 = value1[value1 > VaR]
            ES.append(np.mean(value1))
        else:
            VaR = np.quantile(portfolio, level, interpolation = 'higher') - V
            value2 = portfolio - V
            value2 = value2[value2 > VaR]
            ES.append(np.mean(value2))
        
    result = pd.DataFrame({'MC multi ES':ES}, index = index)
    return result

# Backtest function, parametric and MC ONLY
def Backtesting(port_df, VaR_df, V, period, state):
    N = VaR_df.shape[0]
    n = 252
    v_start = VaR_df.index[0]
    #v_end = VaR_df.index[-1]
    p_start = port_df.index.get_loc(v_start)
    #p_end = port_df.index.get_loc(v_end)
    
    exception = []
    for i in range(N - n - period):
        VaR = np.array(VaR_df.iloc[i:i + n, 0])
        price_change = port_df['Price'].values[p_start + i + period:p_start + i + period + n] \
        - port_df['Price'].values[p_start + i:p_start + i + n]
        Price_change = np.array(price_change) / np.array(port_df['Price'][p_start + i:p_start + i + n]) * V
        
        if state == True:
            count = -Price_change > VaR
            count = np.sum(count)
            exception.append(count)
        else:
            count = Price_change > VaR
            count = np.sum(count)
            exception.append(count)
            
    result = pd.DataFrame({'Exception':exception}, index = VaR_df.index[n - 1:N - period - 1])
    return result

# Stock Info Button: plot portfolio info
def Plot_Position(port_df, stock_df, option_df, stocks, options, int_rate, gbm_df):
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    df = pd.DataFrame()
    if isinstance(stock_df, pd.core.frame.DataFrame):
        df['Portfolio'] = stock_df['Price']
    else:
        for ticker in stocks.keys():
            df[ticker] = stock_df[ticker]['Price']
    
    name = []
    value = []
    
    for ticker in stocks:
        name.append(ticker + ': ' + str(stocks[ticker]))
        value.append(abs(stock_df[ticker].iloc[0, 0] * stocks[ticker]))
    for ticker in options:
        name.append(ticker + ': '+ str(options[ticker][0]))
        iv_index = 3 if options[ticker][1] == 'call' else 0                      # set index for option implied volatility
        iv_lib = dict(zip([3, 6, 12], [1, 2, 3]))
        iv_index = iv_index + iv_lib[options[ticker][2]]
        value.append(abs(float(Black_Scholes(option_df[ticker].iloc[0, 0], 
                                             option_df[ticker].iloc[0, 0], int_rate.values[0], 
                                             option_df[ticker].iloc[0,iv_index], 
                                             options[ticker][2]/12, 
                                             options[ticker][1]) * options[ticker][0])))
    explode = [0.1] * len(name)
    
    fig.suptitle('Visualization of Portfolio Info', fontsize=20)
    ax[0, 0].pie(value, explode=explode, labels=name, shadow=True)
    ax[0, 0].axis('equal')
    ax[0, 0].legend(name, title="Assets", loc="best")
    ax[0, 0].set_title("Initial Asset Allocation")
    if bool(stocks):
        ax[0, 1].plot(df)
        ax[0, 1].set_title('Historical Stock Prices')
        ax[0, 1].legend(df.columns, loc='best')
    ax[1, 0].plot(port_df['Price'])
    ax[1, 0].set_title('Unit Portfolio Value')
    ax[1, 1].plot(gbm_df)
    ax[1, 1].set_title('Portfolio return drift and volatility')
    ax[1, 1].legend(gbm_df.columns, loc='best')
    plt.show()
    #return fig


# Plot VaR Button, 3 different methods VaR and comparison
def Plot_VaR(df, stocks, period, vlevel):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    fig.suptitle('Plot of Portfolio {} day {:.1%} VaR'.format(period, vlevel), fontsize=20)
    if bool(stocks):
        ax[0, 0].plot(df[['Normal VaR', 'GBM VaR']])
        ax[0, 0].legend(df[['Normal VaR', 'GBM VaR']].columns, loc="best")
    else:
        ax[0, 0].plot(df[['GBM VaR']])
        ax[0, 0].legend(df[['GBM VaR']].columns, loc="best")
    ax[0, 0].set_title("Parametric VaR")
    
    ax[0, 1].plot(df[['Historical rel VaR', 'Historical abs VaR']])
    ax[0, 1].set_title('Historical VaR')
    ax[0, 1].legend(df[['Historical rel VaR', 'Historical abs VaR']].columns, loc='best')
    
    ax[1, 0].plot(df[['MC multi VaR', 'MC one VaR']])
    ax[1, 0].set_title('Monte Carlo VaR')
    ax[1, 0].legend(df[['MC multi VaR', 'MC one VaR']].columns, loc='best')
    
    ax[1, 1].plot(df)
    ax[1, 1].set_title('All Methods Comparison')
    ax[1, 1].legend(df.columns, loc='best')
    plt.show()

# Plot ES Button, 2 different methods ES and comparison
def Plot_ES(df, period, elevel): 
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4)    
    fig.suptitle('Plot of Portfolio {} day {:.1%} ES'.format(period, elevel), fontsize=20)

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(df[['Historical rel ES', 'Historical abs ES']])
    ax1.set_title('Historical ES')
    ax1.legend(df[['Historical rel ES', 'Historical abs ES']].columns, loc='best')

    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax2.plot(df[['MC multi ES', 'MC one ES']])
    ax2.set_title('Monte Carlo ES')
    ax2.legend(df[['MC multi ES', 'MC one ES']].columns, loc='best')

    ax3 = fig.add_subplot(gs[2:4, 1:3])
    ax3.plot(df)
    ax3.set_title('All Methods Comparison')
    ax3.legend(df.columns, loc='best')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()

# Backtest Button, plot validation results, return average exception numbers of each method
def Plot_Backtest(loss, exceptions, period, level):
    avg_exceptions = exceptions.mean(axis=0)                                     # average number of exceptions in one year
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2) 
    fig.suptitle('Validation Results of {} day {:.1%} VaR'.format(period, level), fontsize=20)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(loss)
    ax1.set_title('VaR v.s. Actual Loss')
    ax1.set_ylim(bottom=0)
    ax1.legend(loss.columns, loc='best')
    
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(exceptions)
    ax2.set_title('Number of Exceptions')
    ax2.legend(exceptions.columns, loc='best')
    ax2.axhline(y=round(252 * (1 - level), 2), ls='--', color='red')
    plt.show()
    
    return avg_exceptions

# plot number of VaR exceptions
def Plot_exception(bt_result, level):
    expect = round(252 * (1 - level), 2)
    handles = [Line2D([0], [0], color='blue', linestyle='-'), Line2D([0], 
               [0], color='red', linestyle='--')]
    labels = ['number of exceptions', 'expected number = ' + str(expect)]

    plt.plot(bt_result)
    plt.title('1 year VaR exception, p=' + str(level))
    plt.axhline(y=expect, ls='--', color='red')
    plt.legend(handles=handles, labels=labels, loc='best')
    plt.show()


# main read function, output dataframes and adjusted time period
def Read_Main(path, time_start, time_end, l, horizon_period, parameter):
    try:
        option_data = Read_option(path)
        stocks, options = Read_position(path, option_data)
        
        time_start, time_end = Adjust_option_time(option_data, options, time_start, time_end)
        stock_data, time_start, time_end = Read_stock(stocks, time_start, time_end)
        
        if bool(options):
            interest_rate = Read_interest(path, time_start, time_end)
        else:
            interest_rate = None
    
    except (FileNotFoundError, AssertionError, ValueError, NameError, 
            RemoteDataError, OverflowError, KeyError) as err:
        raise ValueError(err)
    
    try:
        p_return, option_data, stock_data, long_p = \
        Portfolio_return(stock_data, option_data, interest_rate, 
                         stocks, options, time_start, time_end)
        rho = Calculate_rho(stock_data, l)
        p_hist = Historical_change(p_return, horizon_period)
        
        if parameter == 'Window':
            gbm_stocks = Calibrate_all(stock_data, l, horizon_period)
            gbm_portfolio = Calibrate(p_return, l, horizon_period)
        elif parameter == 'Exponential':
            gbm_stocks = Calibrate_all_EW(stock_data, l, horizon_period)
            gbm_portfolio = Calibrate(p_return, l, horizon_period)
    
    except ValueError as err:
        raise ValueError(err)
    
    return stocks, options, stock_data, option_data, interest_rate, p_return, long_p, rho, p_hist, \
            time_start, time_end, gbm_stocks, gbm_portfolio

# main calculate function, output four df: VaR, ES, Actual Loss and Exceptions
def Calculate_Main(port_df, port_hist, stock_df, option_df, stocks, options, 
                   rho, int_rate, gbm_stocks, gbm_portfolio, V, l, period, vlevel, elevel, long_p):
    
    gbm_VaR = Parametric_VaR_GBM(gbm_portfolio, V, period, vlevel, long_p)
        
    his_VaR = Historical_VaR(port_hist, V, l, vlevel, long_p)
    MC_one_VaR = MC_VaR_One(gbm_portfolio, V, period, vlevel, 10000, long_p)
    MC_multi_VaR = MC_VaR_Multi(gbm_stocks, port_df, option_df, stock_df, stocks, options, rho, 
                                int_rate, V, l, period, vlevel, 10000, long_p)
    
    His_ES = Historical_ES(port_hist, V, l, elevel, long_p)
    MC_one_ES = MC_ES_One(gbm_portfolio, V, period, elevel, 10000, long_p)
    MC_multi_ES = MC_ES_Multi(gbm_stocks, port_df, option_df, stock_df, stocks, options, rho, 
                              int_rate, V, l, period, elevel, 10000, long_p)
    # combine data
    if bool(stocks):
        normal_VaR = Parametric_VaR_Normal(stock_df, gbm_stocks, stocks, rho, V, l, period, vlevel, long_p)
        VaR_df = pd.concat([MC_multi_VaR, MC_one_VaR, his_VaR, normal_VaR, gbm_VaR], axis=1)
    else:
        VaR_df = pd.concat([MC_multi_VaR, MC_one_VaR, his_VaR, gbm_VaR], axis=1)
    VaR_df = VaR_df.fillna(method='bfill')
    
    ES_df = pd.concat([MC_multi_ES, MC_one_ES, His_ES], axis=1)
    ES_df = ES_df.fillna(method='bfill')
    
    exceptions = pd.DataFrame()
    for name in list(VaR_df.columns):
        Backtest = Backtesting(port_df, VaR_df[[name]], V, period, long_p)
        Backtest.columns = [name]
        exceptions = pd.concat([exceptions, Backtest], axis=1)
    
    v_start = VaR_df.index[0]
    v_end = VaR_df.index[-1]
    p_start = port_df.index.get_loc(v_start)
    p_end = port_df.index.get_loc(v_end)
    
    price_change = port_df['Price'].values[p_start + period:p_end] - port_df['Price'].values[p_start:p_end - period]
    price_change = np.array(price_change) / np.array(port_df['Price'][p_start:p_end - period]) * V
    
    if long_p == True:
        loss = pd.DataFrame({'Actual loss':-price_change}, index = VaR_df.index[:-period-1])
        loss = pd.concat([loss, VaR_df.iloc[:-period, :]], axis=1)
    else:
        loss = pd.DataFrame({'Actual loss':price_change}, index = VaR_df.index[:-period-1])
        loss = pd.concat([loss, VaR_df.iloc[:-period, :]], axis=1)
    
    return VaR_df, ES_df, loss, exceptions

# save calculation results
def Save_Data(VaR, ES, Exceptions, path):
    folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_path = os.path.join(path, folder_name)
    os.mkdir(file_path)
    name1 = os.path.join(file_path, 'VaR.csv')
    VaR = VaR.iloc[::-1]
    VaR.to_csv(name1)
    name2 = os.path.join(file_path, 'ES.csv')
    ES = ES.iloc[::-1]
    ES.to_csv(name2)
    name3 = os.path.join(file_path, 'Exceptions.csv')
    Exceptions = Exceptions.iloc[::-1]
    Exceptions.to_csv(name3)