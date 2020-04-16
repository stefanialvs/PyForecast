from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image

root = Tk()
root.geometry('1280x800')

bg_img = ImageTk.PhotoImage(Image.open('bg.jpg'))
bg_label = Label(image=bg_img)
bg_label.pack()

#titles, labels and buttons
title_1 = Label(root, text='PyForecast', font=("Helvetica Neue Bold", 24), 
                        anchor='w', bg='#010626', fg='white')
data_title = Label(root, text='Data', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray", bg='#010626')
data_label = Label(root, text='Select the data', 
                    font=("Helvetica Neue Light", 16), anchor='w', 
                            bg='#010626', fg='white')
upload_label = Label(root, text='or upload your own', 
                    font=("Helvetica Neue Light", 16), anchor='w', 
                            bg='#010626', fg='white')
data = ['Sample Data 1', 'Sample Data 2']                   
drop_data = ttk.Combobox(root, value=data)

upload_img = ImageTk.PhotoImage(Image.open('upload.png'))
upload_button = Button(root, image=upload_img)

filter_title = Label(root, text='Filters', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray", bg='#010626')
season_label = Label(root, text='Select the seasonality', 
                    font=("Helvetica Neue Light", 16), anchor='w', 
                            bg='#010626', fg='white')

seasons = ['Hourly', 'Daily', 'Weekly', 'Monthly','Yearly']
drop_season = ttk.Combobox(root, value=seasons)


horizon_label = Label(root, text='Input Forecast Horizon', 
                    font=("Helvetica Neue Light", 16), anchor='w', 
                        bg='#010626', fg='white')
horizon = Entry(root, width=15)
model_title = Label(root, text='Models', font=("Helvetica Neue Bold", 18), 
                        anchor='w', fg="gray", bg='#010626')

model_naive_check = Checkbutton(root, text='Naive', bg='#010626', fg='white')
model_seasonalnaive_check = Checkbutton(root, text='SeasonalNaive', 
                                                bg='#010626', fg='white')
model_naive2_check = Checkbutton(root, text='Naive2', bg='#010626', fg='white')
model_rwd_check = Checkbutton(root, text='RandomWalkDrift', bg='#010626', 
                                    fg='white')
model_croston_check = Checkbutton(root, text='Croston', bg='#010626', 
                                        fg='white')
model_movingaverage_check = Checkbutton(root, text='MovingAverage', 
                                                bg='#010626', fg='white')


graph_img = ImageTk.PhotoImage(Image.open('data_graph.png'))
graph_label = Label(root, image=graph_img)
datatable_img = ImageTk.PhotoImage(Image.open('data_table.png'))
datatable_label = Label(root, image=datatable_img)

##### functions placeholder
def selected(event):
    pass

drop_season.bind('<<ComboboxSelected>>', selected)
drop_data.bind('<<ComboboxSelected>>', selected)

def display():
    graph_label.place(x=400, y=130)
    datatable_label.place(x=400, y=450)

analyze_img = ImageTk.PhotoImage(Image.open('analyze.png'))
analyze_button = Button(root, image=analyze_img, border=0, bg='#010626', 
                                command=display)

##### Widget placing

title_1.place(x=50, y=50)
data_title.place(x=50, y=100)
data_label.place(x=50, y=130)
drop_data.place(x=50, y=160)
upload_label.place(x=50, y=190)
upload_button.place(x=50, y=220)
filter_title.place(x=50, y=270)
season_label.place(x=50, y=300)
drop_season.place(x=50, y=330)
horizon_label.place(x=50, y=360)
horizon.place(x=50, y=390)
model_title.place(x=50, y=420)
model_naive_check.place(x=50, y=450)
model_seasonalnaive_check.place(x=50, y=480)
model_naive2_check.place(x=50, y=510)
model_rwd_check.place(x=190, y=450)
model_croston_check.place(x=190, y= 480)
model_movingaverage_check.place(x=190, y=510)

analyze_button.place(x=50, y=600)
root.mainloop()