from tkinter import *
from PIL import ImageTk,Image

root = Tk()
root.geometry('1280x800')

#titles, labels and buttons
title_1 = Label(root, text='PyForecast', font=("Helvetica Neue Bold", 24), 
                        anchor='w')
data_title = Label(root, text='Data', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray")
data_label = Label(root, text='Select the data', 
                    font=("Helvetica Neue Light", 16), anchor='w')
upload_label = Label(root, text='or Upload your own', 
                    font=("Helvetica Neue Light", 16), anchor='w')
clicked1 = StringVar()                    
drop_data = OptionMenu(root, clicked1, 'Sample Data 1', 'Sample Data 2')
upload_button = Button(root, text='Upload New Data')

filter_title = Label(root, text='Filters', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray")
season_label = Label(root, text='Select the seasonality', 
                    font=("Helvetica Neue Light", 16), anchor='w')
clicked2 = StringVar() 
drop_season = OptionMenu(root, clicked2, 'Hourly', 'Daily', 'Weekly', 'Monthly',
                                        'Yearly')
horizon_label = Label(root, text='Input Forecast Horizon', 
                    font=("Helvetica Neue Light", 16), anchor='w')
horizon = Entry(root, width=15)
model_title = Label(root, text='Models', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray")
model_naive_check = Checkbutton(root, text='Naive')
model_seasonalnaive_check = Checkbutton(root, text='SeasonalNaive')
model_naive2_check = Checkbutton(root, text='Naive2')
model_rwd_check = Checkbutton(root, text='RandomWalkDrift')
model_croston_check = Checkbutton(root, text='Croston')
model_movingaverage_check = Checkbutton(root, text='MovingAverage')


analyze_button = Button(root, text='Analyze', font=("Helvetica Neue Bold", 14),
                                bg='blue', fg='black', width=15)


space_1 = Label(root, text=' ')
space_2 = Label(root, text=' ')
space_3 = Label(root, text=' ')
space_4 = Label(root, text=' ')
space_5 = Label(root, text=' ')
space_6 = Label(root, text=' ')

#button and labels positioning
title_1.grid(row=1, column=1)
space_1.grid(row=0, column=0)
space_2.grid(row=2, column=1)
space_5.grid(row=3, column=1)
data_title.grid(row=4, column=1)
data_label.grid(row=5, column=1)
drop_data.grid(row=6, column=1)
upload_label.grid(row=7, column=1)
upload_button.grid(row=8, column=1)
space_2.grid(row=9, column=1)
filter_title.grid(row=10, column=1)
season_label.grid(row=11, column=1)
drop_season.grid(row=12, column=1)
horizon_label.grid(row=13, column=1)
horizon.grid(row=14, column=1)
space_4.grid(row=15, column=1)
model_title.grid(row=16, column=1)
model_naive_check.grid(row=17, column=1)
model_seasonalnaive_check.grid(row=18, column=1)
model_naive2_check.grid(row=19, column=1)
model_rwd_check.grid(row=20, column=1)
model_croston_check.grid(row=21, column=1)
model_movingaverage_check.grid(row=22, column=1)
space_6.grid(row=23, column=1)
analyze_button.grid(row=24, column=1)
root.mainloop()