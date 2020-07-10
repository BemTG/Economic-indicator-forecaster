from flask import Flask, render_template, request , Markup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar, os, io, base64

import datetime
from dateutil import relativedelta
import matplotlib 

import io, base64, os, json, re , math
import warnings
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split







#the render_template allows us to inject python variables into html(webpage)
#request allows us to deal requests from the client side to the webserver
#Markup allows us to inject headers directly to the webpage


#Global Variable
history_cci=None
cci_df=None
history=None
upcoming_forecast=None
cci_df_full=None
model=None
spx_data=None






# Initiate our flask instance
app=application= Flask(__name__)


@application.before_first_request
# load and prepare the data
def Startup():
    
    global val_plot_X,val_plot_y,history,cci_df,upcoming_forecast,cci_df_full,model,spx_data,history_bci,bci_df,upcoming_forecast_bci,bci_df_full,model_bci,history_cli,cli_df,upcoming_forecast_cli,cli_df_full,model_cli,upcoming_forecast_spx,df1,model_spx,history,cci_df_full,spx,Val_X2,Val_y2


    #load the cli data
    cci_raw=pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/OECD.CCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en')
    cci_df=cci_raw[cci_raw['LOCATION'] =='OECD']

    all_spx_data=pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=0&period2=1650067200&interval=1d&event=history')
    spx_data=all_spx_data[['Date','Adj Close']]
    spx_data=spx_data.set_index('Date')
    spx_data=spx_data.fillna(method='ffill')



    full_date_list=[]
    for tm in cci_df['TIME']:
        year=int(tm.split('-')[0])
        month=int(tm.split('-')[1])
        #print calendar full date
        full_date_list.append(tm + '-'+str(calendar.monthrange(year,month)[1]))

    cci_df['Date'] = full_date_list
    cci_df['Date']= pd.to_datetime(cci_df['Date'])
    cci_df=cci_df[['Date','Value']]
    cci_df.index=cci_df['Date']

    cci_df_full=cci_df[['Date','Value']]
    cci_df_full = cci_df_full.set_index('Date')

    #Added learning columns

    prediction_months=12

    #Make a prediction column by shifting the values up by 12 months
    cci_df['Prediction']=cci_df[['Value']].shift(-prediction_months)

    #make several rollingMA to help the model learn better
    cci_df['7_months MA']=cci_df['Value'].rolling(window=7).mean()
    cci_df['4_months MA']=cci_df['Value'].rolling(window=4).mean()
    cci_df['2_months MA']=cci_df['Value'].rolling(window=2).mean()



    #drop the NaN values
    cci_cleaned=cci_df.dropna(subset=['7_months MA','4_months MA','2_months MA', 'Prediction'])

    #Get rid of the NAN values
    cci_cleaned.reset_index(drop=True, inplace=True)
    del cci_cleaned['Date']

    #Get the all the neccessary column values
    values=cci_cleaned[['Value','7_months MA','4_months MA','2_months MA','Prediction']]

    #change values to log form
    values=np.log(values)

    #change format to numpy array
    values=values.to_numpy()


    X=values[:,0:4]
    y=values[:,4:5]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    model=Sequential()
    model.add(Dense(200, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=0)

    model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)
    
    history=model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)

    #Forecast the next dates
    cci_df_full2=cci_df_full.copy()

    #make several rollingMA to help the model learn better
    cci_df_full2['7_months MA']=cci_df_full2['Value'].rolling(window=7).mean()
    cci_df_full2['4_months MA']=cci_df_full2['Value'].rolling(window=4).mean()
    cci_df_full2['2_months MA']=cci_df_full2['Value'].rolling(window=2).mean()

    #drop the NaN values in all columns
    cci_df_full2=cci_df_full2.dropna(subset=['7_months MA','4_months MA','2_months MA'])

    #Reset the index column and delete the date columns
    cci_df_full2.reset_index(drop=True, inplace=True)
    #del cci_df_full2['Date']

    #change values to log form
    cci_df_full2=np.log(cci_df_full2)
    #change format to numpy array
    cci_df_full2=cci_df_full2.to_numpy()
    #Get only the last 12 months datapoints
    cci_df_full2=cci_df_full2[-12:]

    upcoming_forecast= model.predict(cci_df_full2)

    val_plot_X=np.exp(model.predict(X_test))
    val_plot_y=np.exp(y_test)



    #THE BCI

    #load the bli data
    bci_raw=pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/OECD.BCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en')
    bci_df=bci_raw[bci_raw['LOCATION'] =='OECD']

    full_date_list=[]
    for tm in bci_df['TIME']:
        year=int(tm.split('-')[0])
        month=int(tm.split('-')[1])
        #print calendar full date
        full_date_list.append(tm + '-'+str(calendar.monthrange(year,month)[1]))

    bci_df['Date'] = full_date_list
    bci_df['Date']= pd.to_datetime(bci_df['Date'])
    bci_df=bci_df[['Date','Value']]
    bci_df.index=bci_df['Date']

    bci_df_full=bci_df[['Date','Value']]
    bci_df_full = bci_df_full.set_index('Date')

    #Added learning columns

    prediction_months=12

    #Make a prediction column by shifting the values up by 12 months
    bci_df['Prediction']=bci_df[['Value']].shift(-prediction_months)

    #make several rollingMA to help the model learn better
    bci_df['7_months MA']=bci_df['Value'].rolling(window=7).mean()
    bci_df['4_months MA']=bci_df['Value'].rolling(window=4).mean()
    bci_df['2_months MA']=bci_df['Value'].rolling(window=2).mean()



    #drop the NaN values
    bci_cleaned=bci_df.dropna(subset=['7_months MA','4_months MA','2_months MA', 'Prediction'])

    #Get rid of the NAN values
    bci_cleaned.reset_index(drop=True, inplace=True)
    del bci_cleaned['Date']

    #Get the all the neccessary column values
    values=bci_cleaned[['Value','7_months MA','4_months MA','2_months MA','Prediction']]

    #change values to log form
    values=np.log(values)

    #change format to numpy array
    values=values.to_numpy()


    X=values[:,0:4]
    y=values[:,4:5]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    model_bci=Sequential()
    model_bci.add(Dense(200, input_dim=X.shape[1], activation='relu'))
    model_bci.add(Dense(200, activation='relu'))
    model_bci.add(Dense(1))

    model_bci.compile(loss='mean_squared_error', optimizer='adam')
    monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=0)

    model_bci.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)
    
    history_bci=model_bci.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)

    #Forecast the next dates
    bci_df_full2=bci_df_full.copy()

    #make several rollingMA to help the model learn better
    bci_df_full2['7_months MA']=bci_df_full2['Value'].rolling(window=7).mean()
    bci_df_full2['4_months MA']=bci_df_full2['Value'].rolling(window=4).mean()
    bci_df_full2['2_months MA']=bci_df_full2['Value'].rolling(window=2).mean()

    #drop the NaN values in all columns
    bci_df_full2=bci_df_full2.dropna(subset=['7_months MA','4_months MA','2_months MA'])

    #Reset the index column and delete the date columns
    bci_df_full2.reset_index(drop=True, inplace=True)
    #del cci_df_full2['Date']

    #change values to log form
    bci_df_full2=np.log(bci_df_full2)
    #change format to numpy array
    bci_df_full2=bci_df_full2.to_numpy()
    #Get only the last 12 months datapoints
    bci_df_full2=bci_df_full2[-12:]

    upcoming_forecast_bci= model_bci.predict(bci_df_full2)



    #THE CLI
    #load the cli data
    cli_raw=pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/OECD.CLI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en')
    cli_df=cli_raw[cli_raw['LOCATION'] =='OECD']

    full_date_list=[]
    for tm in cli_df['TIME']:
        year=int(tm.split('-')[0])
        month=int(tm.split('-')[1])
        #print calendar full date
        full_date_list.append(tm + '-'+str(calendar.monthrange(year,month)[1]))

    cli_df['Date'] = full_date_list
    cli_df['Date']= pd.to_datetime(cli_df['Date'])
    cli_df=cli_df[['Date','Value']]
    cli_df.index=cli_df['Date']

    cli_df_full=cli_df[['Date','Value']]
    cli_df_full = cli_df_full.set_index('Date')

    #Added learning columns

    prediction_months=12

    #Make a prediction column by shifting the values up by 12 months
    cli_df['Prediction']=cli_df[['Value']].shift(-prediction_months)

    #make several rollingMA to help the model learn better
    cli_df['7_months MA']=cli_df['Value'].rolling(window=7).mean()
    cli_df['4_months MA']=cli_df['Value'].rolling(window=4).mean()
    cli_df['2_months MA']=cli_df['Value'].rolling(window=2).mean()



    #drop the NaN values
    cli_cleaned=cli_df.dropna(subset=['7_months MA','4_months MA','2_months MA', 'Prediction'])

    #Get rid of the NAN values
    cli_cleaned.reset_index(drop=True, inplace=True)
    del cli_cleaned['Date']

    #Get the all the neccessary column values
    values=cli_cleaned[['Value','7_months MA','4_months MA','2_months MA','Prediction']]

    #change values to log form
    values=np.log(values)

    #change format to numpy array
    values=values.to_numpy()


    X=values[:,0:4]
    y=values[:,4:5]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    model_cli=Sequential()
    model_cli.add(Dense(200, input_dim=X.shape[1], activation='relu'))
    model_cli.add(Dense(200, activation='relu'))
    model_cli.add(Dense(1))

    model_cli.compile(loss='mean_squared_error', optimizer='adam')
    monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=0)

    model_cli.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)
    
    history_cli=model_cli.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)

    #Forecast the next dates
    cli_df_full2=cli_df_full.copy()

    #make several rollingMA to help the model learn better
    cli_df_full2['7_months MA']=cli_df_full2['Value'].rolling(window=7).mean()
    cli_df_full2['4_months MA']=cli_df_full2['Value'].rolling(window=4).mean()
    cli_df_full2['2_months MA']=cli_df_full2['Value'].rolling(window=2).mean()

    #drop the NaN values in all columns
    cli_df_full2=cli_df_full2.dropna(subset=['7_months MA','4_months MA','2_months MA'])

    #Reset the index column and delete the date columns
    cli_df_full2.reset_index(drop=True, inplace=True)
    #del cci_df_full2['Date']

    #change values to log form
    cli_df_full2=np.log(cli_df_full2)
    #change format to numpy array
    cli_df_full2=cli_df_full2.to_numpy()
    #Get only the last 12 months datapoints
    cli_df_full2=cli_df_full2[-12:]

    upcoming_forecast_cli= model_cli.predict(cli_df_full2)


    #SPX
    #load the cli data
    cci_raw=pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/OECD.CCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en')
    cci_df=cci_raw[cci_raw['LOCATION'] =='OECD']

    full_date_list=[]
    for tm in cci_df['TIME']:
        year=int(tm.split('-')[0])
        month=int(tm.split('-')[1])
        #print calendar full date
        full_date_list.append(tm + '-'+str(calendar.monthrange(year,month)[1]))

    cci_df['Date'] = full_date_list
    cci_df['Date']= pd.to_datetime(cci_df['Date'])
    cci_df=cci_df[['Date','Value']]
    cci_df.index=cci_df['Date']
    cci_df_full=cci_df[['Date','Value']]
    
    cci_df_full = cci_df_full.set_index('Date')
    
    
    spx=pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=0&period2=1650067200&interval=1d&event=history')
    spx=spx[['Date','Close']]
    spx=spx.set_index('Date')
    spx=spx.fillna(method='ffill')
    
    df1=cci_df_full.join(spx)
    df1=df1.fillna(method='ffill')
    
    
    prediction_months=12

    df=df1.copy()
    #Make a prediction column by shifting the values up by 12 months
    df['Prediction']=df[['Close']].shift(-prediction_months)

    #make several rollingMA to help the model learn better
    df['7_months_CCI_MA']=df['Value'].rolling(window=7).mean()
    df['4_months_CCI_MA']=df['Value'].rolling(window=4).mean()
    df['2_months_CCI_MA']=df['Value'].rolling(window=2).mean()

    df['7_months_c_MA']=df['Close'].rolling(window=7).mean()
    df['4_months_c_MA']=df['Close'].rolling(window=4).mean()
    df['2_months_c_MA']=df['Close'].rolling(window=2).mean()



    #drop the NaN values
    full_df=df.dropna(subset=['7_months_CCI_MA','4_months_CCI_MA','2_months_CCI_MA','7_months_c_MA','4_months_c_MA','2_months_c_MA','Prediction'])

    #cci_cleaned2=cci_svm.dropna(subset=['Prediction'])

    # #Get rid of the NAN values
    full_df.reset_index(drop=True, inplace=True)
    
    
    
    
    future_x_inputs=full_df.copy()

    #drop the 'Prediction' column as that is the y label
    del future_x_inputs['Prediction']
    del future_x_inputs['Close']

    #Normalize the future x inputs and put in array
    future_x_inputs=np.log(future_x_inputs)
    future_x_inputs=future_x_inputs.to_numpy()

    #Bare in mind this does not include the very last 12 months x_feature values 
    # of the cci but the 12 values before it

    #We are only interested in the last 12 month values from the cleaned data
    future_x_inputs=future_x_inputs[-12:]
    #future_x_inputs=np.delete(future_x_inputs,1)
    
    
    all_spx_data=full_df[['Close']]
    all_spx_data.reset_index(drop=True, inplace=True)
    actual_last_12months= np.log(all_spx_data)
    actual_last_12months=actual_last_12months.to_numpy()
    actual_last_12months=actual_last_12months[-12:]
    
    
    #Get the all the neccessary column values
    values=full_df[['Value','7_months_CCI_MA','4_months_CCI_MA','2_months_CCI_MA','7_months_c_MA','4_months_c_MA','2_months_c_MA','Prediction']]

    #change values to log form
    values=np.log(values)

    #change format to numpy array
    values=values.to_numpy()
    
    
    X=values[:,0:7]
    y=values[:,7:8]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    
    
    model_spx=Sequential()

    model_spx.add(Dense(200, input_dim=X.shape[1], activation='relu'))
    model_spx.add(Dense(200, activation='relu'))
    model_spx.add(Dense(1))

    model_spx.compile(loss='mean_squared_error', optimizer='adam')
    monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=0)

    history=model_spx.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],epochs=250)
    
    df_full2=df1.copy()

    #make several rollingMA to help the model learn better
    df_full2['7_months_CCI_MA']=df_full2['Value'].rolling(window=7).mean()
    df_full2['4_months_CCI_MA']=df_full2['Value'].rolling(window=4).mean()
    df_full2['2_months_CCI_MA']=df_full2['Value'].rolling(window=2).mean()

    df_full2['7_months_c_MA']=df_full2['Close'].rolling(window=7).mean()
    df_full2['4_months_c_MA']=df_full2['Close'].rolling(window=4).mean()
    df_full2['2_months_c_MA']=df_full2['Close'].rolling(window=2).mean()

    #drop the NaN values in all columns
    df_full2=df_full2.dropna(subset=['7_months_CCI_MA','4_months_CCI_MA','2_months_CCI_MA','7_months_c_MA','4_months_c_MA','2_months_c_MA'])

    #Reset the index column and delete the date columns
    df_full2.reset_index(drop=True, inplace=True)
    del df_full2['Close']


    #change values to log form
    df_full2=np.log(df_full2)
    #change format to numpy array
    df_full2=df_full2.to_numpy()
    #Get only the last 12 months datapoints
    df_full2=df_full2[-12:]
    
    upcoming_forecast_spx= model_spx.predict(df_full2)
    Val_X2=np.exp(model_spx.predict(X_test))
    Val_y2=np.exp(y_test)











@application.route('/', methods=['POST','GET'])
def GetForecastHome():
	fig0,ax0=plt.subplots(figsize=(16,10))
	plt.plot(val_plot_X)
	plt.plot(val_plot_y)
	ax0.axhline(y=100,color='gray')
	plt.legend(['Model predictions', 'Actual CCI values'],loc='best',fontsize=15)
	plt.ylabel('CCI Values', fontsize=20)
	plt.xlabel('Datapoints',fontsize=20)
	plt.title('Model predictions & actual CCI values ', fontsize=20)
	plt.grid()
	img0= io.BytesIO()
	plt.savefig(img0, format='png')
	img0.seek(0)
	plot_url0=base64.b64encode(img0.getvalue()).decode()

	return render_template('home.html',forecast_plot0=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url0)))
    
    

   



#This is how we allow communication from webpage to GetForecast function
#Handles any traffic that comes from the route
#post is when user is sending request through the header(to the webserver)
#get is through the url
@application.route('/cli', methods=['POST','GET'])
def GetForecast():

    
    

    
    months_out=0
    linewidth=1
    
    #did the client ask for forecast--> quantity of months
    if request.method == 'POST':
        linewidth=5
        #we record how many months of forecast they requested
        months_out= int(request.form['months_out'])
        #If client requested 5 months we will run all the code below
            #i.e make df that has the next 5 moths and do model forecast
    
    #make future df dates
    forwarded_dates=[]
    year=2020
    for i in range(5,13):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    year=2021
    for i in range(1,5):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
        
        
    #append forecasted dates to dataframe
    df_forecast=pd.DataFrame(forwarded_dates)
    df_forecast.columns=['Date']
    
    
    upcoming_f=np.exp(upcoming_forecast_cli)
    upcoming_f=pd.DataFrame(upcoming_f)

    df_forecast['Value']=upcoming_f
    #set column as datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    #set index as date
    df_forecast = df_forecast.set_index('Date')
    
    last_date=cli_df_full.tail(1)
    df_forecast=df_forecast.append(last_date)
    df_forecast=df_forecast.drop_duplicates()
    df_forecast = df_forecast.sort_index()
    
    fig,ax=plt.subplots(figsize=(12,6))
    plt.plot(df_forecast.head(months_out+1), label='Neural net model', linewidth=3)
    plt.plot(cli_df_full, label='Historical CLI values',linewidth=2)
    
    ax.axhline(y=100,color='gray')
    plt.legend(loc='best')
    plt.title('', fontsize=20)

    # plt.xticks(cli_df_full, cli_df_full, fontsize = 7)
    fig.autofmt_xdate()


    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.5))
    plt.grid()
    
    
    #Instead of plot.show()
    img= io.BytesIO()
    plt.savefig(img, format='png')#saving our plot in memory using img(BytesIO) as png
    img.seek(0) # our image is now saved in img
    #encoding and decoding our image to a long string of charachters
    plot_url=base64.b64encode(img.getvalue()).decode()
    


    # fig4,ax4 = plt.subplots(figsize=(16,10))
    # plt.plot(df1['Close'], label='Historical SP500 values', linewidth=linewidth, color='green')

    
    
    # second_axis= ax4.twinx()
    # plt.plot(cli_df['Date'],cli_df['Value'], label='Historical CLI values', color='blue')


    # plt.legend(loc='best')
    # plt.title('CLI and SP500 historical values overlayed together',fontsize=20)
    # plt.grid()
    # second_axis.axhline(y=100,color='gray')
    
    # #Instead of plot.show()
    # img4= io.BytesIO()
    # plt.savefig(img4, format='png')#saving our plot in memory using img(BytesIO) as png
    # img4.seek(0) # our image is now saved in img
    # #encoding and decoding our image to a long string of charachters
    # plot_4=base64.b64encode(img4.getvalue()).decode()

    # Use Markup to inject html code to webpage and pass the encoded image(plot_url)
    return render_template('forecast_cli.html',
        forecast_plot=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url)),
        # forecast_comparison1=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_4)),
        months_out=months_out, 
        default_months_out=months_out
                          )

@application.route('/bci', methods=['POST','GET'])
def GetForecast2():
    

    
    months_out2=0
    linewidth2=1
    
    #did the client ask for forecast--> quantity of months
    if request.method == 'POST':
        linewidth2=5
        #we record how many months of forecast they requested
        months_out2= int(request.form['months_out2'])
        #If client requested 5 months we will run all the code below
            #i.e make df that has the next 5 moths and do model forecast
    
    forwarded_dates=[]
    year=2020
    for i in range(5,13):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    year=2021
    for i in range(1,5):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
        
        
    #append forecasted dates to dataframe
    df_forecast=pd.DataFrame(forwarded_dates)
    df_forecast.columns=['Date']
    
    
    upcoming_f=np.exp(upcoming_forecast_bci)
    upcoming_f=pd.DataFrame(upcoming_f)

    df_forecast['Value']=upcoming_f
    #set column as datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    #set index as date
    df_forecast = df_forecast.set_index('Date')
    
    last_date=bci_df_full.tail(1)
    df_forecast=df_forecast.append(last_date)
    df_forecast=df_forecast.drop_duplicates()
    df_forecast = df_forecast.sort_index()
    
    fig2,ax2=plt.subplots(figsize=(12,6))
    plt.plot(df_forecast.head(months_out2+1), label='Neural net model', linewidth=3)
    plt.plot(bci_df_full, label='Historical BCI values', linewidth=2)
    
    ax2.axhline(y=100,color='gray')
    plt.legend(loc='best')
    plt.title('', fontsize=20)
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, 0.5))
    plt.grid()
    fig2.autofmt_xdate()

    #Instead of plot.show()
    img2= io.BytesIO()
    plt.savefig(img2, format='png')#saving our plot in memory using img(BytesIO) as png
    img2.seek(0) # our image is now saved in img
    #encoding and decoding our image to a long string of charachters
    plot_url2=base64.b64encode(img2.getvalue()).decode()



    #     #plot the existing CLI and SPX
    # fig5,ax5 = plt.subplots(figsize=(16,10))
    # plt.plot(df1['Close'], label='Historical SP500 values', linewidth=linewidth2, color='green')
    
    
    # second_axis= ax5.twinx()
    # plt.plot(bci_df['Date'],bci_df['Value'], label='Historical BCI values', color='blue')

    # plt.legend(loc='best')
    # plt.title('BCI and SP500 historical values overlayed together', fontsize=20)
    # second_axis.axhline(y=100,color='gray')
    
    # #Instead of plot.show()
    # img5= io.BytesIO()
    # plt.savefig(img5, format='png')#saving our plot in memory using img(BytesIO) as png
    # img5.seek(0) # our image is now saved in img
    # #encoding and decoding our image to a long string of charachters
    # plot_5=base64.b64encode(img5.getvalue()).decode()
    

    # Use Markup to inject html code to webpage and pass the encoded image(plot_url)
    return render_template('forecast_bci.html',
        forecast_plot2=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url2)),
        # forecast_plot5=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_5)),
        months_out2=months_out2, 
        default_months_out2=months_out2
                          )


@application.route('/cci', methods=['POST','GET'])
def GetForecast3():
    
    

    
    months_out3=0
    linewidth3=1
    
    #did the client ask for forecast--> quantity of months
    if request.method == 'POST':
        linewidth3=5
        #we record how many months of forecast they requested
        months_out3= int(request.form['months_out3'])
        #If client requested 5 months we will run all the code below
            #i.e make df that has the next 5 moths and do model forecast
    
    #make future df dates
    forwarded_dates=[]
    year=2020
    for i in range(5,13):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    year=2021
    for i in range(1,5):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    
    #append forecasted dates to dataframe
    df_forecast=pd.DataFrame(forwarded_dates)
    df_forecast.columns=['Date']

    upcoming_f=np.exp(upcoming_forecast)
    upcoming_f=pd.DataFrame(upcoming_f)

    df_forecast['Value']=upcoming_f
    #set column as datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    #set index as date
    df_forecast = df_forecast.set_index('Date')
    
    last_date=cci_df_full.tail(1)
    df_forecast=df_forecast.append(last_date)
    df_forecast=df_forecast.drop_duplicates()
    df_forecast = df_forecast.sort_index()



    #plot the existing CLI and forcasted CLI values
    fig3,ax3 = plt.subplots(figsize=(12,6))
    plt.plot(df_forecast.head(months_out3+1), label='Neural net model', linewidth=3)
    plt.plot(cci_df_full, label='Historical CCI values', linewidth=2)
    
    ax3.axhline(y=100,color='gray')
    plt.legend(loc='best')
    plt.title('CCI forecast', fontsize=20)
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, 0.5))
    plt.grid()
    fig3.autofmt_xdate()

    
    
    #Instead of plot.show()
    img3= io.BytesIO()
    plt.savefig(img3, format='png')#saving our plot in memory using img(BytesIO) as png
    img3.seek(0) # our image is now saved in img
    #encoding and decoding our image to a long string of charachters
    plot_url3=base64.b64encode(img3.getvalue()).decode()



    # fig6,ax6 = plt.subplots(figsize=(16,10))
    # plt.plot(df1['Close'], label='Historical SP500 values', linewidth=linewidth3, color='green')
    
    
    # second_axis= ax6.twinx()
    # plt.plot(cci_df['Date'],cci_df['Value'], label='Historical CCI values', color='blue')

    # plt.legend(loc='best')
    # plt.title('CCI and SP500 historical values overlayed together', fontsize=20)
    # second_axis.axhline(y=100,color='gray')
    
    # #Instead of plot.show()
    # img6= io.BytesIO()
    # plt.savefig(img6, format='png')#saving our plot in memory using img(BytesIO) as png
    # img6.seek(0) # our image is now saved in img
    # #encoding and decoding our image to a long string of charachters
    # plot_6=base64.b64encode(img6.getvalue()).decode()


    
    

    # Use Markup to inject html code to webpage and pass the encoded image(plot_url)
    return render_template('forecast_cci.html',
        forecast_plot3=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url3)),
        # forecast_plot6=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_6)),
        months_out3=months_out3, 
        default_months_out3=months_out3
                          )

@application.route('/spx', methods=['POST','GET'])
def GetForecast4():
    
    

    
    months_out4=0
    linewidth4=1
    
    #did the client ask for forecast--> quantity of months
    if request.method == 'POST':
        linewidth4=5
        #we record how many months of forecast they requested
        months_out4= int(request.form['months_out4'])
        #If client requested 5 months we will run all the code below
            #i.e make df that has the next 5 moths and do model forecast
    
    #make future df dates
    forwarded_dates=[]
    year=2020
    for i in range(5,13):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    year=2021
    for i in range(1,5):
        forwarded_dates.append(str(year)+'-'+str(i)+'-'+str(calendar.monthrange(year,i)[1]))
    
    #append forecasted dates to dataframe
    df_forecast=pd.DataFrame(forwarded_dates)
    df_forecast.columns=['Date']

    upcoming_f=np.exp(upcoming_forecast_spx)
    upcoming_f=pd.DataFrame(upcoming_f)

    df_forecast['Close']=upcoming_f
    #set column as datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    #set index as date
    df_forecast = df_forecast.set_index('Date')
    
    last_date=df1['Close'].tail(1)
    last_date=pd.DataFrame(last_date)
    df_forecast=df_forecast.append(last_date)
    df_forecast=df_forecast.drop_duplicates()
    df_forecast = df_forecast.sort_index()



    #plot the existing CLI and forcasted CLI values
    fig4,ax4 = plt.subplots(figsize=(12,6))
    plt.plot(df_forecast.head(months_out4+1), label='Neural net model', linewidth=3)
    plt.plot(df1['Close'], label='Historical SP500 values', linewidth=2)
    
    
    plt.legend(loc='best')
    # plt.title('SPX Forecast  ')
    plt.grid()
    fig4.autofmt_xdate()

    
    
    #Instead of plot.show()
    img4= io.BytesIO()
    plt.savefig(img4, format='png')#saving our plot in memory using img(BytesIO) as png
    img4.seek(0) # our image is now saved in img
    #encoding and decoding our image to a long string of charachters
    plot_url4=base64.b64encode(img4.getvalue()).decode()



    # fig6,ax6=plt.subplots(figsize=(8,5))
    # plt.plot(Val_X2)
    # plt.plot(Val_y2 )
    # plt.legend(['Model prediction', 'Actual SP500 values'],loc='best',fontsize=6)
    # plt.ylabel('CCI Values', fontsize=9)
    # plt.xlabel('Datapoints',fontsize=9)
    # plt.title('Validation results for SP500 neural net model', fontsize=9)
    # plt.grid()
    # img6= io.BytesIO()
    # plt.savefig(img6, format='png')
    # img6.seek(0)
    # plot_url6=base64.b64encode(img6.getvalue()).decode()


    return render_template('spx.html',
        forecast_plot4=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url4)),
        # forecast_plot6=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url6)),
        months_out4=months_out4, 
        default_months_out4=months_out4
                          )

if __name__ =='__main__':
    application.run(host='0.0.0.0',port=80, debug=True)
    #host='0.0.0.0',port=80, debug=True
    