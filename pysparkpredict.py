import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType
import random
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Linear Regression with Synthetic Dataset") \
    .getOrCreate()

# Function to generate random data for each feature
def generate_synthetic_data(n_rows):
    # Generate sample data for each feature
    locations = ["City A", "City B", "City C", "City D"]
    room_types = ["Single", "Double", "Suite"]
    amenities = ["Wi-Fi", "Pool", "Gym", "Spa"]
    customer_ratings = [float(i) for i in range(1, 6)]
    local_events = ["Concert", "Festival", "Exhibition", "Sports"]
    time_of_year = ["Spring", "Summer", "Autumn", "Winter"]
    price_range = [float(i) for i in range(50, 301, 50)]

    # Generate synthetic data
    data = []
    for _ in range(n_rows):
        data.append((
            locations[random.randint(0, len(locations) - 1)],
            room_types[random.randint(0, len(room_types) - 1)],
            amenities[random.randint(0, len(amenities) - 1)],
            customer_ratings[random.randint(0, len(customer_ratings) - 1)],
            local_events[random.randint(0, len(local_events) - 1)],
            time_of_year[random.randint(0, len(time_of_year) - 1)],
            price_range[random.randint(0, len(price_range) - 1)]
        ))

    # Create DataFrame from the generated data
    schema = ["Location", "Room_Type", "Amenities", "Customer_Ratings", "Local_Events", "Time_of_Year", "Price"]
    df = spark.createDataFrame(data, schema)
    return df

# Generate synthetic dataset with 100 rows
synthetic_dataset = generate_synthetic_data(n_rows=100)

#sytheticdata = synthetic_dataset.write.csv("C:\Users\IndraKiranReddy\Desktop\Projects\SyntheticDataset\synthetic_dataset.csv", header=True, mode="overwrite")


# Convert categorical features to numerical using StringIndexer and OneHotEncoder
string_indexer_cols = ["Location", "Room_Type", "Amenities", "Local_Events", "Time_of_Year"]
one_hot_encoder_cols = [col + "_index" for col in string_indexer_cols]

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in string_indexer_cols]
encoder = OneHotEncoder(inputCols=one_hot_encoder_cols, outputCols=[col + "_encoded" for col in string_indexer_cols])

# Assemble the features into a single vector column
feature_cols = ["Customer_Ratings"] + encoder.getOutputCols()
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Create the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="Price")

# Create a pipeline to chain all the preprocessing steps and the linear regression model
pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])

# Split the dataset into training and testing sets (80% training, 20% testing)
(training_data, testing_data) = synthetic_dataset.randomSplit([0.8, 0.2], seed=123)

# Train the linear regression model
model = pipeline.fit(training_data)

# Make predictions on the testing set
predictions = model.transform(testing_data)

# Show the predictions and actual prices
predictions.select("features", "Price", "prediction").show()
#predictions.write.json("C:\\Users\\IndraKiranReddy\Desktop\\Projects\\SyntheticDataset\\predictions_dataset.json" ,mode="overwrite")

# Stop the Spark session
#spark.stop()


# Initialize Flask app
app = Flask(__name__)

# Create a Spark session
spark = SparkSession.builder \
    .appName("Hotel Price Prediction API") \
    .getOrCreate()

# Load the trained model and other preprocessing steps (use the same pipeline as before)
# ... (include the relevant code from the previous sections for preprocessing and model training)

# Endpoint for predicting room price
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Get input data in JSON format from the request
        input_data = request.get_json()

        # Convert JSON data to a DataFrame
        input_df = spark.createDataFrame([input_data])

        # Make predictions using the trained model
        predictions = model.transform(input_df)

        # Extract the predicted price
        predicted_price = predictions.select("prediction").collect()[0][0]

        # Return the predicted price as JSON response
        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(debug=True)
