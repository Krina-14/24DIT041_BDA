import os
import sys

# Ensure Java 17 is used if available (prevents Java 24 compatibility issues with Hadoop)
if os.path.exists("/opt/homebrew/opt/openjdk@17"):
    os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def main():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("E-Commerce Sales Analysis") \
        .getOrCreate()

 

    # Define explicit schema to ensure OrderID is read as String and numeric columns as Integer
    schema = StructType([
        StructField("OrderID", StringType(), True),
        StructField("Product", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("Price", IntegerType(), True),
        StructField("City", StringType(), True)
    ])

    # Load sales data CSV with defined schema
    csv_path = "sales_data.csv"
    df = spark.read.csv(csv_path, header=True, schema=schema)

    # 1. Original Data
    print("\n" + "="*45)
    print("===== ORIGINAL DATA =====")
    print("="*45)
    df.show(truncate=False)

    # 2. Schema
    print("\n" + "="*45)
    print("===== SCHEMA =====")
    print("="*45)
    df.printSchema()

    # 3. Data Types
    print("\n" + "="*45)
    print("===== DATA TYPES =====")
    print("="*45)
    for col_name, dtype in df.dtypes:
        print(f"Column: {col_name:<10} -> Data Type: {dtype}")

    # 4. Revenue Calculation
    print("\n" + "="*45)
    print("===== REVENUE CALCULATION =====")
    print("="*45)
    df = df.withColumn("Revenue", col("Quantity") * col("Price"))
    df.show(truncate=False)

    # 5. Category-wise Revenue
    print("\n" + "="*45)
    print("===== CATEGORY-WISE REVENUE =====")
    print("="*45)
    category_revenue = df.groupBy("Category") \
        .agg(sum("Revenue").alias("Total_Revenue")) \
        .orderBy(col("Total_Revenue").desc())
    category_revenue.show(truncate=False)

    # 6. Top-Selling Products
    print("\n" + "="*45)
    print("===== TOP-SELLING PRODUCTS =====")
    print("="*45)
    top_products = df.groupBy("Product") \
        .agg(sum("Quantity").alias("Total_Quantity_Sold")) \
        .orderBy(col("Total_Quantity_Sold").desc())
    top_products.show(truncate=False)

    # 7. City-wise Revenue
    print("\n" + "="*45)
    print("===== CITY-WISE REVENUE =====")
    print("="*45)
    city_revenue = df.groupBy("City") \
        .agg(sum("Revenue").alias("Total_Revenue")) \
        .orderBy(col("Total_Revenue").desc())
    city_revenue.show(truncate=False)

    # 8. Top 3 Revenue-Generating Cities
    print("\n" + "="*45)
    print("===== TOP 3 REVENUE-GENERATING CITIES =====")
    print("="*45)
    top_3_cities = city_revenue.limit(3)
    top_3_cities.show(truncate=False)

    # 9. Average Revenue Per Order
    print("\n" + "="*45)
    print("===== AVERAGE REVENUE PER ORDER =====")
    print("="*45)
    avg_revenue = df.select(avg("Revenue").alias("Average_Revenue_Per_Order"))
    avg_revenue.show(truncate=False)

    # 10. Discount Analysis (10% Discount)
    print("\n" + "="*45)
    print("===== DISCOUNT ANALYSIS =====")
    print("="*45)
    df_discount = df.withColumn("Discount", col("Revenue") * 0.10) \
                    .withColumn("Final_Revenue", col("Revenue") - col("Discount"))
    df_discount.show(truncate=False)

    # 11. Least-Performing Category
    print("\n" + "="*45)
    print("===== LEAST-PERFORMING CATEGORY =====")
    print("="*45)
    least_performing_cat = df_discount.groupBy("Category") \
        .agg(sum("Final_Revenue").alias("Total_Final_Revenue")) \
        .orderBy(col("Total_Final_Revenue").asc()) \
        .limit(1)
    least_performing_cat.show(truncate=False)

    # 12. Monthly Revenue
    print("\n" + "="*45)
    print("===== MONTHLY REVENUE =====")
    print("="*45)
    print("Monthly revenue analysis requires an OrderDate column.")
    print("The current dataset does not contain a date column.")

    spark.stop()

if __name__ == "__main__":
    main()
