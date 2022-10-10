import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import pickle
## importing necessery library
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
@st.cache
def download_stock_data(stock_list):
    curr_date = datetime.datetime.now()
    prev_date = curr_date - datetime.timedelta(days=90)
    period_1 = int(prev_date.timestamp())
    period_2 = int(curr_date.timestamp())
    params ={
    'period1': period_1, #this is timestamp for date 01/01/2014 20:38:12
    'period2': period_2, #this is timestamp for date 01/01/2018 20:38:11
    'events' : 'history'
     }
    stock_data = []
    for stock in stock_list:
        #user agent is an identity string which helps the server to identify about its clients
        # use your own user agent !
        response = requests.get(stock_url.format(stock),params = params, headers = 
    {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'})
        response = response.text
        response = response.split('\n')
      
        for row in response:
            row = row.split(',')
            #we need to drop the 1st row of each stock,because it contains header information
            if row!= ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] :
                #adding the stock name / ticker, since it is not mentioned in response from request
             
                stock_data.append(row)
        
        #now let's convert into pandas dataframe
    stock_data = pd.DataFrame(stock_data, columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
	#stock_data.set_index('Date',drop = True,inplace = True)
  
    #stock_data will be a list of data, so we need to convert it into pandas dataframe in future
	#stock_data['Ticker'] = sp500_ind['Ticker'].apply(lambda x : 'SP500_Ind' if x == '%5EGSPC' else x)
    return stock_data

#Since, date column is of object type, we need to convert it into Datetime type
#sp500_ind['Date'] = pd.to_datetime(sp500_ind['Date'])
#sp500_ind.set_index('Date',drop = True,inplace = True)
#Date = st.slider(sp500_ind['Date'])

def price_plot(df):
  df['Date'] = df.index
  fig = plt.figure()
  plt.plot(df.Open, color='skyblue', alpha=0.8)
  plt.title(df, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Opening Price', fontweight='bold')
  return st.pyplot(fig)
  
def main():
	st.title("SP500 Stock Price Prediction")
	
	menu = ['Home', 'plot_data']
	choice = st.sidebar.selectbox("Menu", menu)
	
	if choice == 'Home':
		st.subheader('Home')
	elif choice == 'plot_data':
		st.subheader('plot_data')
	return choice
	
if __name__ == '__main__':
	choice = main()
	
	
if choice == 'Home':
	"""In this project we will predict future stock price of SP500 index , using previous 63 days info which include 'Open','High','Low','Close','Volume',
	A very imp point to note is that, this prediction model should not be consider as a solely baseline  to invest your hard money to invest in stock, please
	make your own research before investment"""

elif choice == 'plot_data':
	#we will download our data from yahoo finance url
	stock_url = "https://query1.finance.yahoo.com/v7/finance/download/{}"
	
	sp500_ind = download_stock_data(['%5EGSPC'])
	data_load_state = st.text('Loading data...')
	data_load_state.text("Done! (using st.cache)")
	st.write(sp500_ind.head(63))
	
	#time_series = sp500_ind['Open']
	st.write('This is a line_chart.')
	#with open(r"C:\Users\DELL\streamlit_data_app\model.pkl","rb") as pickle_in:
	    #model = pickle.load(pickle_in)
	#price_plot(sp500_ind)
	#filename = 'finalized_model.sav'
	#loaded_model = pickle.load(open(filename, 'rb'))
	scaler_x = pickle.load(open('scaler_x.sav','rb'))
	scaler_y = pickle.load(open('scaler_y.sav','rb'))
	loaded_model = load_model('model.h5')
	
	X = sp500_ind[['Open','High','Low','Close','Volume']].tail(63).values
	X_scaled=scaler_x.fit_transform(np.array(X))
	
	
	#input = np.array([[[0.05307898, 0.05290937, 0.08757889, 0.07622034, 0.23964458],
	#[0.05590343, 0.05259162, 0.08925955, 0.07648994, 0.4647416 ],
	#[0.06040262, 0.06080094, 0.0965659 , 0.08442616, 0.22934918],
	#[0.06362875, 0.05954586, 0.09456066, 0.07800611, 0.24703956],
	#[0.05911374, 0.05619785, 0.09229647, 0.07745127, 0.26781798],
	#[0.05835387, 0.05595563, 0.09327401, 0.07701749, 0.25306415],
	#[0.05725991, 0.05586817, 0.09243948, 0.07850237, 0.46399697],
	#[0.06070893, 0.05945055, 0.09472677, 0.07880326, 0.22635087],
	#[0.05793227, 0.05537173, 0.08790725, 0.07110926, 0.26421605],
	#[0.05489299, 0.05649174, 0.09124555, 0.07943623, 0.25686185],
	#[0.06033492, 0.05639642, 0.08262171, 0.07123433, 0.30341316],
	#[0.05509189, 0.05241689, 0.08878432, 0.07268401, 0.24063975],
	#[0.05596306, 0.05498252, 0.09150439, 0.07487226, 0.07757968],
	#[0.05572045, 0.05328665, 0.08893893, 0.07625163, 0.24129581],
	#[0.05280054, 0.04906888, 0.08339445, 0.06734635, 0.24098118],
	#[0.04885435, 0.05011733, 0.08564316, 0.07337565, 0.18700577],
	#[0.05322622, 0.05213493, 0.08902392, 0.07425484, 0.19840238],
	#[0.05435601, 0.05106262, 0.08536112, 0.07351245, 0.21094331],
	#[0.05769753, 0.05759982, 0.0942322 , 0.08043658, 0.21850259],
	#[0.06137324, 0.05901373, 0.09652723, 0.08222626, 0.20638933],
	#[0.06303205, 0.0646613 , 0.09845909, 0.08669646, 0.16781448],
	#[0.06714538, 0.06437536, 0.10250439, 0.08664561, 0.17440191],
	#[0.06570528, 0.0636207 , 0.0998694 , 0.0872201 , 0.19409078],
	#[0.06887587, 0.06874807, 0.10508929, 0.09238585, 0.20546758],
	#[0.07353414, 0.07025332, 0.10685881, 0.09223731, 0.21980306],
	#[0.07028799, 0.06823573, 0.10555678, 0.09188176, 0.20544543],
	#[0.07213386, 0.06845816, 0.10604362, 0.09085403, 0.19969702],
	#[0.07445697, 0.07169098, 0.10936249, 0.09367526, 0.32762571],
	#[0.07528842, 0.07186968, 0.10937409, 0.09394886, 0.26341316],
	#[0.07639831, 0.07280304, 0.10357465, 0.09300708, 0.31451145],
	#[0.0709723 , 0.06862891, 0.10540218, 0.09170985, 0.23285323],
	#[0.07368526, 0.07038829, 0.10689748, 0.09100648, 0.25318068],
	#[0.07414677, 0.07060675, 0.10790586, 0.09337055, 0.25220766],
	#[0.07545148, 0.07134938, 0.1061054 , 0.09384726, 0.25427256],
	#[0.07372109, 0.0696218 , 0.10702115, 0.09173322, 0.27366777],
	#[0.07405917, 0.07119851, 0.10826912, 0.09355811, 0.22590456],
	#[0.0741626 , 0.07157579, 0.10973726, 0.09515231, 0.1905541 ],
	#[0.07464403, 0.07551566, 0.1075891 , 0.09281171, 0.23866457],
	#[0.06947257, 0.06897836, 0.1044054 , 0.09246007, 0.23440191],
	#[0.06948442, 0.06539201, 0.09500495, 0.07846717, 0.27087689],
	#[0.05980193, 0.05852523, 0.09504362, 0.07968246, 0.21714036],
	#[0.06533933, 0.06651998, 0.10165439, 0.08926373, 0.17782555],
	#[0.07078923, 0.06679002, 0.10422382, 0.08878311, 0.18837732],
	#[0.07077728, 0.06918489, 0.10509316, 0.09015075, 0.1931026 ],
	#[0.06851778, 0.06524909, 0.09201443, 0.07526301, 0.21512323],
	#[0.0544713 , 0.05541943, 0.08841342, 0.07352027, 0.24694284],
	#[0.05362004, 0.05157099, 0.087123  , 0.07462222, 0.17381693],
	#[0.05690192, 0.06117822, 0.09342852, 0.08405496, 0.17257472],
	#[0.06132944, 0.05885083, 0.0964229 , 0.08074529, 0.17348366],
	#[0.06253479, 0.05943862, 0.09440218, 0.07876415, 0.18062693],
	#[0.06126582, 0.0608565 , 0.09673202, 0.08035845, 0.1505844 ],
	#[0.0623121 , 0.05893432, 0.09549951, 0.08082342, 0.16094622],
	#[0.05618187, 0.05896205, 0.09131507, 0.0816284 , 0.16792402],
	#[0.06179891, 0.06337849, 0.09733091, 0.08604002, 0.15581425],
	#[0.06839842, 0.06921669, 0.10462557, 0.09153394, 0.23906893],
	#[0.07308059, 0.07134938, 0.10895301, 0.09344868, 0.1647952 ],
	#[0.07146159, 0.06800932, 0.09840504, 0.08614162, 0.25563363],
	#[0.0688679 , 0.06708391, 0.1032926 , 0.08914649, 0.24213366],
	#[0.07055061, 0.0666789 , 0.10371378, 0.08897458, 0.23974713],
	#[0.06823934, 0.06607918, 0.10336986, 0.08754045, 0.23375284],
	#[0.0731204 , 0.07475305, 0.10921185, 0.09796581, 0.23250248],
	#[0.08005012, 0.07785885, 0.11533588, 0.10123637, 0.22541281],
	#[0.08082582, 0.07849434, 0.11601967, 0.10197494, 0.24139253]]])
	X_up_scaled=scaler_x.inverse_transform(np.array(X_scaled))
	train_predict = loaded_model.predict(np.array([X_scaled]))
	y_pred = scaler_y.inverse_transform(train_predict)
	#Price tomorrow = Price today * (Return + 1)
	pred_price = X_up_scaled[-1][0] * (y_pred +1)
	st.write(pred_price)
	
