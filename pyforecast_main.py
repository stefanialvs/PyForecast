from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
from pandastable import Table
import time


from src.utils_ml import ml_pipeline, plot_grid

def labels():
    global root
    root = Tk()
    root.geometry('1440x900')
    
    # Background
    global bg_img 
    bg_img = ImageTk.PhotoImage(Image.open('./ui/bgw.png'))
    bg_label = Label(image=bg_img)
    bg_label.pack()

    # Interface Titles
    title_1 = Label(root, text='PyForecast', font=("Helvetica Neue Bold", 36), 
                            anchor='w', fg='black').place(x=600, y=20)
    data_title = Label(root, text='Data', font=("Helvetica Neue Bold", 18), 
                            anchor='w', fg="#666666").place(x=50, y=80)
    data_label = Label(root, text='Select the data', 
                        font=("Helvetica Neue Regular", 13), 
                        anchor='w', fg='black').place(x=50, y=120)

    #Filter labels and buttons
    filter_title = Label(root, text='Filters', font=("Helvetica Neue Bold", 18), 
                            anchor='w', fg="#666666").place(x=50, y=200)
    season_label = Label(root, text='Select the frequency', 
                        font=("Helvetica Neue Regular", 13), 
                         anchor='w', fg='black').place(x=50, y=240)
    horizon_label = Label(root, text='Input Forecast Horizon', 
                    font=("Helvetica Neue Regular", 13), 
                          anchor='w', fg='black').place(x=50, y=300)
    model_title = Label(root, text='Models', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="#666666").place(x=50, y=380)
    metric_title = Label(root, text='Metrics', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="#666666").place(x=50,y=540)
    model_sample_title = Label(root, text='Models Sample', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="#666666").place(x=390,y=80)
    metric_table_title = Label(root, text='Evaluations', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="#666666").place(x=720,y=540)
    residuals_plot_title = Label(root, text='Residuals', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="#666666").place(x=390,y=540)
    global series_img
    series_img = ImageTk.PhotoImage(Image.open('./ui/series_blank.jpg'))
    series_blank = Label(image=series_img)
    series_blank.place(x=350, y=150)
    
    global progress
    progress = ttk.Progressbar(root, orient='horizontal', length=200, 
                                    mode='determinate')
    progress.place(x=530, y=90, height=10)

    
def model_buttons():
    #Model Parameters for Checkboxes
    model_widgets = {'Naive': {'bool': IntVar(), 'place': (50, 420)},
                     'SeasonalNaive': {'bool': IntVar(), 'place': (50, 445)}, 
                     'Naive2': {'bool': IntVar(), 'place': (50, 470)},
                     'RandomWalkDrift': {'bool': IntVar(), 'place': (170, 420)}, 
                     'Croston': {'bool': IntVar(), 'place': (170, 445)},
                     'MovingAverage': {'bool': IntVar(), 'place': (170, 470)},
                     'ESRNN': {'bool': IntVar(), 'place': (50, 495)},
                     'SeasonalMovingAverage': {'bool': IntVar(), 'place': (170, 495)}}
    
    # Declare check buttons for each model
    for model_name in model_widgets:
        model_widgets[model_name]["check_button"] = Checkbutton(root, text=model_name, 
                                                                fg='black', font=("Helvetica Neue Regular", 13),
                                                                variable=model_widgets[model_name]['bool'])
    #Placing check buttons for each model
    for model_name in model_widgets:
        x, y = model_widgets[model_name]["place"]
        check_button = model_widgets[model_name]["check_button"]
        check_button.place(x=x, y=y)
    
    return model_widgets
    

def metric_buttons():
    #Metric Parameters for Checkboxes
    metric_widgets = {'RMSE': {'bool': IntVar(), 'place':(50, 585)},
                      'MAPE': {'bool': IntVar(), 'place': (50, 610)},
                      'SMAPE': {'bool': IntVar(), 'place': (50, 635)},
                      'MASE': {'bool': IntVar(), 'place': (170, 585)},
                      'RMSSE': {'bool': IntVar(), 'place': (170, 610)}}

    # Declare check buttons for each metric
    for metric_name in metric_widgets:
        metric_widgets[metric_name]["check_button"] = Checkbutton(root, text=metric_name,
                                                                  fg='black', font=("Helvetica Neue Regular", 13),
                                                                  variable=metric_widgets[metric_name]['bool'])
    # Place check buttons for each metric
    for metric_name in metric_widgets:
        x, y = metric_widgets[metric_name]["place"]
        check_button = metric_widgets[metric_name]["check_button"]
        check_button.place(x=x, y=y)
    
    return metric_widgets

def buttons():
    #Data selection button
    data = ['./data/m4/Hourly.csv',
            './data/m4/Daily.csv',
            './data/m4/Weekly.csv',
            './data/m4/Monthly.csv',
            './data/m4/Quarterly.csv',
            './data/m4/Yearly.csv']

    global data_widget
    data_widget = ttk.Combobox(root, value=data, width=18)
    data_widget.place(x=50, y=150)
    
    #Frequency button
    global freq_widget
    freqs = ['Hourly', 'Daily', 'Weekly', 'Monthly','Quarterly', 'Yearly']
    freq_widget = ttk.Combobox(root, value=freqs, width=18)
    freq_widget.place(x=50, y=270)
    
    #Horizon entry
    global horizon_widget
    horizon_widget = Entry(root, width=18)
    horizon_widget.place(x=50, y=330)
    
    global model_widgets
    model_widgets = model_buttons()
    
    global metric_widgets
    metric_widgets = metric_buttons()
    
    global analyze_img
    analyze_img = ImageTk.PhotoImage(Image.open('./ui/analyze.png'))
    analyze_button = Button(root, image=analyze_img, border=0, 
                            command=analyze,  padx=0, pady=0)
    analyze_button.config(highlightbackground='white')
    analyze_button.place(x=50, y=685)
    
    global new_sample_img
    new_sample_img = ImageTk.PhotoImage(Image.open('./ui/new_sample.png'))
    sample_button = Button(root, image=new_sample_img, border=0, 
                            command=display_grid,  padx=0, pady=0)
    sample_button.config(highlightbackground='white')
    sample_button.place(x=390, y=115)

def analyze():
    global h
    directory, h, freq, models_filter, metrics_filter = get_run_parameters()
    
    global y_df, models
    y_df, models = ml_pipeline(directory=directory, 
                               h=h, freq=freq, 
                               models_filter=models_filter,
                               metrics_filter=metrics_filter,
                               progress_bar=progress)
    
    # Plot grid
    display_grid()
    
    # Evaluations table
    ## Documentation from:
    #https://pandastable.readthedocs.io/en/latest/examples.html
    frame = tk.Frame(root, background='white')
    frame.place(x=720,y=580)
    pt = Table(frame, width=600, height=130)
    pt.importCSV('./results/metrics.csv')
    pt.show()
    
    global residuals_img
    residuals_img = ImageTk.PhotoImage(Image.open('./results/residuals_distribution.png'))
    residuals_label = Label(image=residuals_img)
    residuals_label.place(x=350, y=580)

def display_grid():
    plot_grid(y_df, models, h)
    global plot_file
    plot_file = ImageTk.PhotoImage(Image.open('./results/grid_series.png'))
    plot_label = Label(image=plot_file)
    plot_label.place(x=350, y=150)

def progress_bar():
    progress['maximum']=100
    for i in range(101):
        time.sleep(0.05)
        progress['value']=i
        progress.update()
    
##### Selection Functions

def directory_selection():
    return data_widget.get()

def horizon_selection():
    h = int(horizon_widget.get())
    return h

def freq_selection():
    return freq_widget.get()

def model_selection():
    models_filter = []
    for model_name in model_widgets:
        model_bool = model_widgets[model_name]['bool']
        if model_bool.get() == 1:
            models_filter.append(model_name)
    return models_filter
        
def metric_selection():
    metrics_filter = []
    for metric_name in metric_widgets:
        metric_bool = metric_widgets[metric_name]['bool']
        if metric_bool.get() == 1:
            metrics_filter.append(metric_name)
    return metrics_filter

def get_run_parameters():
    directory = directory_selection()
    h = horizon_selection()
    freq = freq_selection()
    models_filter = model_selection()
    metrics_filter = metric_selection()
    return directory, h, freq, models_filter, metrics_filter

#ttk.Progressbar
#bg='#010626'

labels()
buttons()
root.mainloop()