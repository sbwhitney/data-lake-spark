import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Create an Apache Spark session to process the data.
    
    Args:
    
    * N/A
    
    Output:
    
    * spark -- An Apache Spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Load JSON input data (song_data) from input_data path,
        process the data to extract songs_table and artists_table, and
        store the queried data to parquet files.
        
    Args:
    
    * spark: reference to Spark session
    * input_data: path to input_data to be processed (song_data)
    * output_data: path to location to store the output (parquet files)
    
    Output:
    
    * songs_table: directory with parquet files stored in output_data path
    * artists_table: directory with parquet files stored in output_data path
    """
    # define json structure
    songdata_schema = StructType([
        StructField("song_id", StringType(), True),
        StructField("year", StringType(), True),
        StructField("duration", DoubleType(), True),
        StructField("artist_id", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("artist_location", StringType(), True),
        StructField("artist_latitude", DoubleType(), True),
        StructField("artist_longitude", DoubleType(), True),
    ])
    
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data, schema=songdata_schema)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'artist_id', 'year', 'duration')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(output_data + "songs")

    # extract columns to create artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location', 'artist_latitude',
                              'artist_longitude')
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists")


def process_log_data(spark, input_data, output_data):
    """Load JSON input data (log_data) from input_data path,
        process the data to extract users_table, time_table,
        songplays_table, and store the queried data to parquet files.
        
    Args:
    
    * spark: reference to Spark session
    * input_data: path to input_data to be processed (log_data)
    * output_data: path to location to store the output (parquet files)
                          
    Output:
    
    * users_table: directory with users_table parquet files 
                   stored in output_data path
    * time_table: directory with time_table parquet files
                  stored in output_data path
    * songplays_table: directory with songplays_table parquet files
                       stored in output_data path
    """
    # define json structure
    logdata_schema = StructType([
        StructField("artist", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("itemInSession", LongType(), True),
        StructField("lastName", StringType(), True),
        StructField("length", DoubleType(), True),
        StructField("level", StringType(), True),
        StructField("location", StringType(), True),
        StructField("method", StringType(), True),
        StructField("page", StringType(), True),
        StructField("registration", DoubleType(), True),
        StructField("sessionId", LongType(), True),
        StructField("song", StringType(), True),
        StructField("status", LongType(), True),
        StructField("ts", LongType(), True),
        StructField("userAgent", StringType(), True),
        StructField("userId", StringType(), True),
    ])
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data, schema=logdata_schema)

    # filter by actions for song plays
    df = df.filter(col("page") == 'NextSong')

    # extract columns for users table    
    users_table = df.select(col("userId").alias("user_id"),col("firstName").alias("first_name"),
                            col("lastName").alias("last_name"),"gender","level")
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x/1000)), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select(col("timestamp").alias("start_time"),
                                   hour(col("timestamp")).alias("hour"),
                                   dayofmonth(col("timestamp")).alias("day"), 
                                   weekofyear(col("timestamp")).alias("week"), 
                                   month(col("timestamp")).alias("month"),
                                   year(col("timestamp")).alias("year"))
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data + "time")

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, "song_data/A/A/A/TRAAAAK128F9318786.json")
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.join(df, 
        song_df.artist_name==df.artist).withColumn("songplay_id", 
            monotonically_increasing_id()).withColumn("start_time", get_datetime(df.ts)).select("songplay_id",
               "start_time",                         
               col("userId").alias("user_id"),
               "level",
               "song_id",
               "artist_id",
               col("sessionId").alias("session_id"),
               col("artist_location").alias("location"),
               "userAgent",
               month(col("start_time")).alias("month"),
               year(col("start_time")).alias("year"))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(output_data + "songplays")


def main():
    """Load JSON input data (song_data and log_data) from input_data path,
        process the data to extract songs_table, artists_table,
        users_table, time_table, songplays_table,
        and store the queried data to parquet files to output_data path.
        
    Args:
    
    * NA
    
    Output:
    
    * songs_table: directory with songs_table parquet files
                   stored in output_data path
    * artists_table: directory with artists_table parquet files
                     stored in output_data path
    * users_table: directory with users_table parquet files
                   stored in output_data path
    * time_table: directory with time_table parquet files
                  stored in output_data path
    * songplays_table: directory with songplays_table parquet files
                       stored in output_data path
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-nanodegree-data-engineer/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
