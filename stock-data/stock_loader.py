import pandas as pd
import yfinance as yh

class stock_data_loader:
    def __init__(self,ticker:str,start_date:str = "2020-01-01",end_date:str = None):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
        self.data = None
        
    def download_data(self):
        print(f"Downloading {self.ticker} from {self.start_date} to {self.end_date}")
        self.data = yh.download(self.ticker,start=self.start_date,end=self.end_date)
        return self.data
    
    def Clean_data(self):
        if self.data is None:
            raise ValueError("Call download_data() first")
        df = self.data.dropna() 
        df.columns = [col[0].lower().replace(" " ,"_") for col in df.columns]
        df.index = pd.to_datetime(df.index)
        
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        
        self.data = df
        
    def add_features(self):
        if self.data is None: 
         raise ValueError("Data not available. Call download_data() and clean_data() first.")
        if 'adj_close' not in self.data.columns:
          raise ValueError("adj_close column missing. Run clean_data() first.")
      
        df = self.data
        
        #daily returns
        df["returns"] =  df["adj_close"].pct_change() * 100
        
        #20-day moving average
        df["ma_20"] = df["adj_close"].rolling(window=20).mean()
        
        #20-day volatility
        df["volatility"] = df["returns"].rolling(window=20).std()
        
        df.dropna(inplace=True)
        self.data = df
    
    def get_latest_n_days(self , n:int):
        return self.data.tail(n)
    
    def save_to_csv(self, filepath:str):
        self.data.to_csv(filepath)
        print(f"data saved to {filepath}")
        
    def save_to_sqlite(self , db_path:str,table_name:str=None):
        import sqlite3
        if table_name is None:
            table_name = f"stock_{self.ticker.lower()}"
        conn = sqlite3.connect(db_path)
        self.data.to_sql(table_name,conn,if_exists="replace",index=True)
        conn.close()
        print(f"data saved to {db_path} in table {table_name}")
        

        
    
