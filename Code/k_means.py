from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import GroupedData
import configparser

sc.stop()

#Initialisation de la session Spark
spark=SparkSession.builder.appName('kmeans').getOrCreate()
sc=spark.sparkContext
sqlContext = SQLContext(sc)

#Récuperer les path depuis le fichier de configuration
config = configparser.ConfigParser()
config.read(r"C:/Users/sakur/OneDrive/Documents/Cours/Outils Data Mining/K-means/properties.ini")

path_to_input_data= config.get('Bristol_City_bike','Input_data')
path_to_output_data= config.get('Bristol_City_bike','Output_data')
num_partition_kmeans = config.get('Bristol_City_bike','Kmeans_level') 

#Importation du json avec Spark
bristol = sqlContext.read.json(path_to_input_data)

#Création du nouveau DataFrame 
Kmeans_df=bristol.select(F.col('longitude'),F.col('latitude'))

#K-means
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
num_partition_kmeans = float(num_partition_kmeans)
features = ('longitude','latitude')
kmeans = KMeans().setK(num_partition_kmeans).setSeed(1)
assembler = VectorAssembler(inputCols=features,outputCol="features")
dataset=assembler.transform(Kmeans_df)
model = kmeans.fit(dataset)
fitted = model.transform(dataset)

#Nom des colonnes de fitted
fitted.show()

#Longitudes et latitudes moyennes de chaque cluster
fitted.groupBy(fitted.prediction).agg({"latitude":"avg","longitude":"avg"}).show()

#Exportation du DataFrame sous format CSV après élimination de la colonne features
fitted = fitted.drop("features")
path=path_to_output_data+"fitted_model.csv"
fitted.toPandas().to_csv(path,index=False)

sc.stop()
