#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!pip install pandas numpy mysql-connector-python sqlalchemy openpyxl


# In[2]:


import os
import pandas as pd
import numpy as np
import logging
import mysql.connector
from sqlalchemy import create_engine


# In[3]:


logging.basicConfig(filename='etl_pipeline.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# In[4]:


def create_database():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="Ashwin@2005"  # ✅ Replace with your actual password
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS sales;")
        print("✅ Database 'sales' created or already exists.")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to create database:", e)


# In[5]:


os.makedirs("data", exist_ok=True)

sample_data = {
    "Store_ID": ["S001", "S002", "S001"],
    "Date": ["2025-07-18", "2025-07-18", "2025-07-18"],
    "Product_ID": ["P001", "P002", "P003"],
    "Product_Name": ["Shampoo", "Soap", "Toothpaste"],
    "Quantity_Sold": [10, 15, 5],
    "Unit_Price": [120.0, 40.0, 90.0],
    "Discount_Percent": [10.0, 5.0, 0.0],
    "Payment_Mode": ["Cash", "Card", "UPI"]
}

df_sample = pd.DataFrame(sample_data)
df_sample.to_csv("data/test_sales.csv", index=False)
print("✅ Sample CSV created at: data/test_sales.csv")


# In[6]:


def extract_data(folder_path='data/'):
    all_data = []
    try:
        files = os.listdir(folder_path)
        print("🔍 Found files:", files)
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, file))
                all_data.append(df)

        if not all_data:
            raise ValueError("❌ No CSV files found in the folder!")

        combined_df = pd.concat(all_data, ignore_index=True)
        logging.info("Extraction complete with %d rows.", len(combined_df))
        return combined_df
    except Exception as e:
        logging.error("Extraction failed: %s", str(e))
        raise


# In[7]:


def transform_data(df):
    try:
        df.dropna(inplace=True)
        df["Total_Sale_Value"] = df["Quantity_Sold"] * df["Unit_Price"] * (1 - df["Discount_Percent"] / 100)
        df.columns = [col.lower() for col in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df.drop_duplicates(subset=["store_id", "date", "product_id"], inplace=True)

        conditions = [
            (df["total_sale_value"] >= 1000),
            (df["total_sale_value"] >= 500) & (df["total_sale_value"] < 1000),
            (df["total_sale_value"] < 500)
        ]
        choices = ["High", "Medium", "Low"]
        df["sale_category"] = np.select(conditions, choices, default="Unknown")

        logging.info("Transformation complete.")
        return df
    except Exception as e:
        logging.error("Transformation failed: %s", str(e))
        raise


# In[8]:


def load_to_mysql(df):
    try:
        # NOTE: Use encoded '@' as '%40' in password
        db_url = "mysql+mysqlconnector://root:Ashwin%402005@127.0.0.1:3306/sales"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            df.to_sql('retail_sales', con=conn, if_exists='replace', index=False)
        logging.info("Load to MySQL successful.")
        print("✅ Data loaded into MySQL table 'retail_sales'")
    except Exception as e:
        logging.error("MySQL Load failed: %s", str(e))
        raise


# In[9]:


def analyze_and_report(df):
    try:
        total_sales = df.groupby("store_id")["total_sale_value"].sum().reset_index()
        top_products = df.groupby("product_name")["total_sale_value"].sum().sort_values(ascending=False).head(5)
        daily_trend = df.groupby(["date", "store_id"])["total_sale_value"].sum().unstack().fillna(0)

        with pd.ExcelWriter("store_sales_summary.xlsx") as writer:
            total_sales.to_excel(writer, sheet_name="Total Sales Per Store", index=False)
            top_products.to_frame().to_excel(writer, sheet_name="Top 5 Products")
            daily_trend.to_excel(writer, sheet_name="Daily Trend")

        logging.info("Reporting complete.")
        print("📊 Analysis exported to 'store_sales_summary.xlsx'")
    except Exception as e:
        logging.error("Reporting failed: %s", str(e))
        raise


# In[10]:


def run_etl_pipeline():
    logging.info("🚀 ETL Pipeline started.")
    create_database()
    df_raw = extract_data()
    df_clean = transform_data(df_raw)
    load_to_mysql(df_clean)
    analyze_and_report(df_clean)
    logging.info("✅ ETL Pipeline completed successfully.")


# In[11]:


run_etl_pipeline()


# In[15]:


#get_ipython().system('jupyter nbconvert --to script "Retail sales.ipynb"')


# In[ ]:




