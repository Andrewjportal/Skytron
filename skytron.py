#!/usr/bin/env python
# coding: utf-8

# In[1]:


from opensky_api import OpenSkyApi
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from bokeh.layouts import row
from bokeh.plotting import figure, curdoc
from bokeh.driving import linear
from bokeh.models import Range1d, HoverTool
from bokeh.client import push_session
import random
from threading import Thread
import time
import pprint
from sklearn.externals import joblib
import pickle


api = OpenSkyApi()
sky_dict={}
dict_sky={}
ANOMALY_DICT={}
roll_avg_error_dict={}
model = MLPRegressor()



def fix_dicts():
    global sky_dict
    global dict_sky
    
    sd_keys=set(sky_dict.keys())
    ds_keys=set(dict_sky.keys())

    if not sd_keys.issubset(ds_keys):
        mis_keys = sd_keys-ds_keys
        for mis_key in mis_keys:
            dict_sky[f'{mis_key}']=(0,0,0,0,0)
            
    if not ds_keys.issubset(sd_keys):
        mis_keys = ds_keys-sd_keys
        for mis_key in mis_keys:
            sky_dict[f'{mis_key}']=(0,0,0,0,0)


sky_dict={}
states= api.get_states()

for s in states.states:
    vector=(s.longitude, s.latitude, s.geo_altitude, s.velocity, s.heading)
    vector=[0 if x==None else x for x in vector]
    sky_dict[f'{s.icao24}']= vector



def stream_learn():
    global sky_dict
    global dict_sky
    global roll_avg_error_dict
     #Create dict for anomaly_dict at timestamp
    global ANOMALY_DICT
    global model
    while True: 

        time.sleep(10)
        states = api.get_states()
        #get another dict
        for s in states.states:
            vector=(s.longitude, s.latitude, s.geo_altitude, s.velocity, s.heading)
            vector=[0 if x==None else x for x in vector]
            dict_sky[f'{s.icao24}']=vector


        fix_dicts()

        sd_keys=set(sky_dict.keys())
        ds_keys=set(dict_sky.keys())

        
        for key in sd_keys:
            X=np.asarray(sky_dict[key]).reshape(1,-1)
            Y=np.asarray(dict_sky[key]).reshape(1,-1)
            model.partial_fit(X,Y)

        error_dict={}
        for key in sd_keys:
            X=np.asarray(sky_dict[key]).reshape(1,-1)
            Y=np.asarray(dict_sky[key]).reshape(1,-1)
            y=model.predict(X)
            error_dict[key]=mean_squared_error(Y,y)



        time_error_dict={}
        time_error_dict[states.time]=error_dict

        
        roll_avg_error_dict[states.time]=np.mean(list(error_dict.values()))

        roll_std_error_dict={}
        roll_std_error_dict[states.time]=np.std(list(error_dict.values()))

        roll_avg_error_dict[states.time]=(np.mean(list(error_dict.values()))+list(roll_avg_error_dict.values())[-1])/2
        roll_std_error_dict[states.time]=(np.std(list(error_dict.values()))+list(roll_std_error_dict.values())[-1])/2

        std_error=roll_std_error_dict[states.time]

       
        #create dict for anomae
        anomaly_dict={}
        for key in error_dict:
            if error_dict[key]>std_error:
                anomaly_dict[key]=error_dict[key]
        ANOMALY_DICT[states.time]=anomaly_dict

        sky_dict=dict_sky
    


t1 = Thread(target=stream_learn)
t1.start()
time.sleep(10)    


# In[ ]:

def plot_stuff():
    #Plot1   cumalitive erro
    hover =[("Seconds", "@x"),("Avg Error", "@y")]
    p = figure(plot_width=400, plot_height=400,tooltips=hover,title="Real-Time Error")

    #p.x_range.follow="end"
    #p.x_range.follow_interval = 20
    p.x_range.range_padding=0
    p.xaxis.axis_label = "Time"
    p.xaxis.major_label_text_font_size = '4pt'

    #p.y_range.follow="end"
    #p.y_range.follow_interval = 20
    #p.y_range.range_padding=0
    p.y_range = Range1d(0, 3500)
    p.yaxis.axis_label = "Rolling AVG Error"

    r1 = p.line([], [], color="firebrick", line_width=2)

    ds1 = r1.data_source



        
    #Plot 2 counting anomalies
    hover2 =[("Seconds", "@x"),("Anomalies", "@y")]
    p2 = figure(plot_width=400, plot_height=400,tooltips=hover2,title="Anomaly Detection")

    #p2.x_range.follow="end"
    #p2.x_range.follow_interval = 10
    p2.x_range.range_padding=0
    p2.xaxis.axis_label = "Time"
    p2.xaxis.major_label_text_font_size = '4pt'

    #p.y_range.follow="end"
    #p.y_range.follow_interval = 20
    #p.y_range.range_padding=0
    p2.y_range = Range1d(0, 500)
    p2.yaxis.axis_label = "Anomaly Count"

    r2 = p2.line([], [], color="#1D91C0", line_width=2)

    ds2 = r2.data_source

    d=row(p, p2)

    session = push_session(curdoc(),session_id="skytron")


    @linear()
    def update(step):
        global roll_avg_error_dict
        
        current_error=list(roll_avg_error_dict.values())[-1]
        ds2.data['x'].append(step*10)
        ds2.data['y'].append(current_error)
        ds2.trigger('data', ds2.data, ds2.data,)

        global ANOMALY_DICT
        error_d=list(ANOMALY_DICT.values())[-1]
        ds1.data['x'].append(step*10)
        ds1.data['y'].append(len(error_d.values()))
        ds1.trigger('data', ds1.data, ds1.data)


        

    curdoc().add_root(d)
    curdoc().add_periodic_callback(update, 10000)

    # open the doc in browser
    session.show()

    # run forever
    session.loop_until_closed()

t2 = Thread(target=plot_stuff)
t2.start()




    
def get_user_input():
        global ANOMAlY_DICT
        global model
       
        while True:
            user_input=input("Enter Command ")
            
            if user_input=='T':
              
                pprint.pprint(list(ANOMALY_DICT.values())[-1])
            elif user_input=='save':
                joblib.dump(model, 'skytron.pkl')
                joblib.dump(ANOMALY_DICT, 'ANOMALY_DICT.pkl')
                
            

            elif user_input=='x':
               break

            else:
                print("No work; type x to escape")
        

t3 = Thread(target=get_user_input)
t3.start()
    





