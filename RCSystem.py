# python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 02:15:50 2019

@author: yz3380@columbia.edu
"""

import os
import pandas as pd
import tkinter as tk
import tkinter.messagebox
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk
from my_utils import *


# GUI setups
AUTHOR_INFO = 'Version 1.3\nYunxiao Zhao, Ophelia Jiang\nyz3380@columbia.edu, wj2289@columbia.edu'
LABEL_NAME_1 = ['Starting Time:', 'Ending Time:', 'Portfolio Value:', 'Est. Method:']
LABEL_NAME_2 = ['VaR Level:', 'ES Level:', 'Window Length:', 'Horizon Period:']
SMALL = ('Arial', 13)
MEDIUM = ('Arial', 16)
WIDTH = 10


def Errormsg(msg):                                                               # global errmsgbox function
    tkinter.messagebox.showwarning("Error", msg)

def Noticemsg(msg):
    tkinter.messagebox.showinfo("Notification", msg)


class RMSystem(tk.Tk):                                                           # base class

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("GR5320 Project v1.3")
        #tk.Tk.wm_title(self, 'GR5320 Project v1.0')
        self.geometry("640x480")
        
        container = tk.Frame(self)
        container.pack()
        
        self.frames = {}
        
        for F in (IndexPage, CalPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        
        self.show_frame(IndexPage)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class IndexPage(tk.Frame):                                                       # index page
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        s = ttk.Style()                                                          # set ttk button styles
        s.configure('TButton', font=('Arial', 18))
        s.configure('my.TButton', font=('Arial', 16))
        s.configure('Red.TButton', font=('Arial', 18), foreground="red")
 
        Name = tk.Label(self, text="Risk Calculation System", font=("Arial", 32, 'bold'), fg="blue")
        Name.grid(row=0, column=0, rowspan=3, columnspan=4, sticky="NWES", ipadx=12, pady=20)
        
        Menu = tk.Label(self, text="Select a function:", font=("Arial", 12))
        Menu.grid(row=3, column=1, columnspan=2, sticky="S", pady=10)
        
        GoWin = ttk.Button(self, text="Start", style = 'TButton', width=15, 
                            command=lambda: controller.show_frame(CalPage))
        GoWin.grid(row=4, column=1, columnspan=2, ipadx=20, ipady=10, pady=5)
        
        GoMac = ttk.Button(self, text="Help", style = 'TButton', width=15, 
                              command=lambda: self.Open_Help())
        GoMac.grid(row=5, column=1, columnspan=2, ipadx=20, ipady=10, pady=5)
        
        Quit = ttk.Button(self, text="Quit", style='Red.TButton', width=15, command=lambda: quit())
        Quit.grid(row=6, column=1, columnspan=2, ipadx=20, ipady=10, pady=5)
        
        Author = tk.Label(self, text=AUTHOR_INFO, font=("Arial", 8), fg="grey")
        Author.grid(row=7, column=1, columnspan=2, sticky="NWES", pady=20)
        
        self.grid_rowconfigure(3, minsize=75)
    
    def Open_Help(self):
        url = "https://github.com/yz3380/5320Project"
        webbrowser.open(url, new=1)

class CalPage(tk.Frame):                                                         # calculation page
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
                                                             # initialize variables
        self.path = os.getcwd()
        self.ready_to_plot = False
        
        self.Time_start = pd.Timestamp('1890-8-9')
        self.Time_end = pd.Timestamp('2077-7-7')
        self.VaR_p = 0.99
        self.ES_p = 0.975
        self.Initial = 10000.0
        self.Length = 5.0
        self.Horizon = 5
        self.Method = 'Window'
        
        Widgets = {}                                                             # initialize input widgets
        
        Instruc = tk.Label(self, text="Please enter valid inputs:", font=SMALL)
        Instruc.grid(row=0, column=0, columnspan=2, pady=10, sticky="WS")
        
        for i, parameter in enumerate(LABEL_NAME_1):
            label = ttk.Label(self, text=parameter, font=SMALL, width=WIDTH+3, 
                              foreground='blue', anchor="e")
            label.grid(row=i + 1, column=0, pady=10)
        
        for i, parameter in enumerate(LABEL_NAME_2):
            label = ttk.Label(self, text=parameter, font=SMALL, width=WIDTH+3, 
                              foreground='blue', anchor="e")
            label.grid(row=i + 1, column=2, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.StringVar(), width=WIDTH, bd=5)
        entry.insert(0, '{:%Y-%m-%d}'.format(self.Time_start))
        Widgets["start"] = entry
        entry.grid(row=1, column=1, padx=5, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.StringVar(), width=WIDTH, bd=5)
        entry.insert(0, '{:%Y-%m-%d}'.format(self.Time_end))
        Widgets["end"] = entry
        entry.grid(row=2, column=1, padx=5, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.DoubleVar(), width=WIDTH, bd=5)
        entry.delete(0, tk.END)
        entry.insert(0, self.Initial)
        Widgets["initial"] = entry
        entry.grid(row=3, column=1, padx=5, pady=10)
        
        entry = tk.Listbox(self, selectmode="SINGLE", font=SMALL, width=WIDTH+4, bd=3, height=2)
        entry.insert(1, "Window")
        entry.insert(2, "Exponential")
        entry.select_set(0)
        entry.configure(exportselection=False)
        Widgets["method"] = entry
        entry.grid(row=4, column=1, padx=5, pady=5)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.DoubleVar(), width=WIDTH, bd=5)
        entry.delete(0, tk.END)
        entry.insert(0, self.VaR_p)
        Widgets["vlevel"] = entry
        entry.grid(row=1, column=3, padx=5, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.DoubleVar(), width=WIDTH, bd=5)
        entry.delete(0, tk.END)
        entry.insert(0, self.ES_p)
        Widgets["elevel"] = entry
        entry.grid(row=2, column=3, padx=5, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.DoubleVar(), width=WIDTH, bd=5)
        entry.delete(0, tk.END)
        entry.insert(0, self.Length)
        Widgets["length"] = entry
        entry.grid(row=3, column=3, padx=5, pady=10)
        
        entry = tk.Entry(self, font=MEDIUM, textvar=tk.IntVar(), width=WIDTH, bd=5)
        entry.delete(0, tk.END)
        entry.insert(0, self.Horizon)
        Widgets["period"] = entry
        entry.grid(row=4, column=3, padx=5, pady=5)
                                                                                 # Intialize function buttons
        Clear_Btn = ttk.Button(self, text="Default", style='TButton', width=8, 
                              command=lambda: self.Clear(Widgets))
        Clear_Btn.grid(row=5, column=0, ipadx=5, ipady=5, pady=10, sticky = 's')
        
        Cal_Btn = tk.Button(self, text="Calculate", font=('Arial', 20, 'bold'), width=8, 
                            fg="white", bg="light green", command=lambda: self.Main(Widgets))
        Cal_Btn.grid(row=5, column=3, pady=10)
        
        Port_Btn = ttk.Button(self, text="Portfolio", style='TButton', width=8, 
                            command=lambda: self.Plot_Position())
        Port_Btn.grid(row=6, column=0, ipadx=5, ipady=5)
        
        VaR_Btn = ttk.Button(self, text="Plot VaR", style='TButton', width=8, 
                            command=lambda: self.Plot_VaR())
        VaR_Btn.grid(row=6, column=1, ipadx=5, ipady=5)
        
        ES_Btn = ttk.Button(self, text="Plot ES", style='TButton', width=8, 
                            command=lambda: self.Plot_ES())
        ES_Btn.grid(row=6, column=2, ipadx=5, ipady=5)
        
        Bt_Btn = ttk.Button(self, text="Backtest", style='TButton', width=8, 
                            command=lambda: self.Plot_Backtest())
        Bt_Btn.grid(row=6, column=3, ipadx=5, ipady=5)
        
        Back_Btn = ttk.Button(self, text="Back", style='TButton', width=8, 
                            command=lambda: controller.show_frame(IndexPage))
        Back_Btn.grid(row=7, column=0, ipadx=5, ipady=5, pady=10)
        
        Save_Btn = ttk.Button(self, text="Save", style='TButton', width=8, 
                            command=lambda: self.Save_Data())
        Save_Btn.grid(row=7, column=2, ipadx=5, ipady=5, pady=10)
        
        Quit_Btn = ttk.Button(self, text="Quit", style='Red.TButton', width=8, 
                            command=lambda: quit())
        Quit_Btn.grid(row=7, column=3, ipadx=5, ipady=5, pady=10)
        
    def Clear(self, widgets):
        widgets['start'].delete(0, tk.END)
        widgets['start'].insert(0, '1890-8-9')
        widgets['end'].delete(0, tk.END)
        widgets['end'].insert(0, '2077-7-7')
        widgets['vlevel'].delete(0, tk.END)
        widgets['vlevel'].insert(0, 0.99)
        widgets['elevel'].delete(0, tk.END)
        widgets['elevel'].insert(0, 0.975)
        widgets['initial'].delete(0, tk.END)
        widgets['initial'].insert(0, 10000.0)
        widgets['length'].delete(0, tk.END)
        widgets['length'].insert(0, 5.0)
        widgets['period'].delete(0, tk.END)
        widgets['period'].insert(0, 5)
        widgets['method'].selection_clear(0, 2)
        widgets['method'].select_set(0)     
    
    def Main(self, widgets):                                                     # main function
        
        try:
            assert(widgets['method'].curselection()!=())
        except AssertionError:
            Errormsg('Please select an estimation method!')
            return
        
        try:
            self.Time_start = pd.Timestamp(widgets['start'].get())
            self.Time_end = pd.Timestamp(widgets['end'].get())
            self.VaR_p = float(widgets['vlevel'].get())
            self.ES_p = float(widgets['elevel'].get())
            self.Initial = float(widgets['initial'].get())
            self.Length = float(widgets['length'].get())
            self.Horizon = int(widgets['period'].get())
            self.Method = widgets['method'].get(widgets['method'].curselection())
                                                                                 # check input validity
            Check_input(self.Time_start, self.Time_end, self.VaR_p, 
                        self.ES_p, self.Initial, self.Length, self.Horizon, self.Method)

        except (ValueError, AssertionError):
            Errormsg('Invalid inputs! Please check your setups')
            return
        
        try:
            self.Stocks, self.Options, self.Stock_Data, self.Option_Data, \
            self.Interest_Rate, self.P_Return, self.Long_P, self.Rho, self.P_Hist, \
            self.Time_start, self.Time_end, self.GBM_Stocks, self.GBM_Portfolio = \
            Read_Main(self.path, self.Time_start, self.Time_end, 
                      self.Length, self.Horizon, self.Method)
        except ValueError as err:
            Errormsg(err)
            return
        
        self.VaR, self.ES, self.Loss, self.Exceptions = \
        Calculate_Main(self.P_Return, self.P_Hist, self.Stock_Data, 
                       self.Option_Data, self.Stocks, self.Options, 
                       self.Rho, self.Interest_Rate, self.GBM_Stocks, 
                       self.GBM_Portfolio, self.Initial, self.Length, 
                       self.Horizon, self.VaR_p, self.ES_p, self.Long_P)
        
        self.ready_to_plot = True
        widgets['start'].delete(0, tk.END)
        widgets['start'].insert(0, '{:%Y-%m-%d}'.format(self.Time_start))
        widgets['end'].delete(0, tk.END)
        widgets['end'].insert(0, '{:%Y-%m-%d}'.format(self.Time_end))
        
        Noticemsg('Calculation complete!\nTime period automatically adjusted to\n{} ~ {}.'
                  .format(self.Time_start.strftime('%Y-%m-%d'), self.Time_end.strftime('%Y-%m-%d')))
        
        if (self.P_Return['Price'].min()==0.01 or self.P_Return['Price'].max()==-0.01):
            Errormsg('WARNING:  Portfolio value sign changes during the period, \nresults may not be reliable!!!')
    
    def Plot_Position(self):
        if not self.ready_to_plot:
            Errormsg('Please calculate result first')
            return
        port_df, stock_df, option_df, stocks, options, int_rate, gbm_df = \
        self.P_Return, self.Stock_Data, self.Option_Data, self.Stocks, \
        self.Options, self.Interest_Rate, self.GBM_Portfolio
    
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
            iv_index = 3 if options[ticker][1] == 'call' else 0                  # implied volatility index
            iv_lib = dict(zip([3, 6, 12], [1, 2, 3]))
            iv_index = iv_index + iv_lib[options[ticker][2]]
            value.append(abs(float(Black_Scholes(option_df[ticker].iloc[0, 0], 
                                                 option_df[ticker].iloc[0, 0], 
                                                 int_rate.values[0], 
                                                 option_df[ticker].iloc[0,iv_index], 
                                                 options[ticker][2]/12, options[ticker][1]) 
                                   * options[ticker][0])))
        
        explode = [0.1] * len(name)
        
        root2 = tk.Toplevel()
        root2.title('Portfolio Info')
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        canvas2 = FigureCanvasTkAgg(fig, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
    
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
        ax[1, 1].set_title('Portfolio Return Drift and Volatility')
        ax[1, 1].legend(gbm_df.columns, loc='best')
        
        toobar = NavigationToolbar2Tk(canvas2, root2)
        toobar.update()
        canvas2.draw()
    
    def Plot_VaR(self):
        if not self.ready_to_plot:
            Errormsg('Please calculate result first')
            return
        df, stocks, period, vlevel = self.VaR, self.Stocks, self.Horizon, self.VaR_p
        
        root2 = tk.Toplevel()
        root2.title('Analysis of VaR')
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        canvas2 = FigureCanvasTkAgg(fig, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
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
        
        toobar = NavigationToolbar2Tk(canvas2, root2)
        toobar.update()
        canvas2.draw()     
    
    def Plot_ES(self):
        if not self.ready_to_plot:
            Errormsg('Please calculate result first')
            return
        df, period, elevel = self.ES, self.Horizon, self.ES_p
        
        root2 = tk.Toplevel()
        root2.title('Analysis of ES')
        fig = plt.figure(figsize=(12, 8))
        
        canvas2 = FigureCanvasTkAgg(fig, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        
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
        
        toobar = NavigationToolbar2Tk(canvas2, root2)
        toobar.update()
        canvas2.draw()
             
    def Plot_Backtest(self):
        if not self.ready_to_plot:
            Errormsg('Please calculate result first')
            return
        loss, exceptions, period, level = self.Loss, self.Exceptions, self.Horizon, self.VaR_p

        avg_exceptions = exceptions.mean(axis=0)                                 # average number of exceptions in one year
        Noticemsg("The {} method has the lowest average number of exceptions: {:.2f} days in one year."
                  .format(avg_exceptions.idxmin(), min(avg_exceptions)))

        root2 = tk.Toplevel()
        root2.title('Validation Result')
        fig = plt.figure(figsize=(12, 8))
        canvas2 = FigureCanvasTkAgg(fig, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        
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
        
        toobar = NavigationToolbar2Tk(canvas2, root2)
        toobar.update()
        canvas2.draw()
     
    def Save_Data(self):
        if not self.ready_to_plot:
            Errormsg('Please calculate result first')
            return
        Save_Data(self.VaR, self.ES, self.Exceptions, self.path)
        Noticemsg("Results successfully saved")
    
        
def quit():
    app.destroy()
    plt.close('all')


app = RMSystem()                                                                 # run app in the center of the screen. 
positionRight = int(app.winfo_screenwidth()/2 - 320)
positionDown = int(app.winfo_screenheight()/2 - 240)
app.geometry("+{}+{}".format(positionRight, positionDown))
app.mainloop()
