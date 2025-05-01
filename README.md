# ML-Project
# Airbnb Price Prediction



!pip install xgboost
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
Requirement already satisfied: xgboost in c:\users\lap\anaconda3\lib\site-packages (2.1.4)
Requirement already satisfied: numpy in c:\users\lap\anaconda3\lib\site-packages (from xgboost) (1.26.4)
Requirement already satisfied: scipy in c:\users\lap\anaconda3\lib\site-packages (from xgboost) (1.13.1)
# Load dataset
df = pd.read_csv("C:/Users/lap/Downloads/Airbnb_data - airbnb_data.csv")
df.head()
id	log_price	property_type	room_type	amenities	accommodates	bathrooms	bed_type	cancellation_policy	cleaning_fee	...	latitude	longitude	name	neighbourhood	number_of_reviews	review_scores_rating	thumbnail_url	zipcode	bedrooms	beds
0	6901257	5.010635	Apartment	Entire home/apt	{"Wireless Internet","Air conditioning",Kitche...	3	1.0	Real Bed	strict	True	...	40.696524	-73.991617	Beautiful brownstone 1-bedroom	Brooklyn Heights	2	100.0	https://a0.muscache.com/im/pictures/6d7cbbf7-c...	11201	1.0	1.0
1	6304928	5.129899	Apartment	Entire home/apt	{"Wireless Internet","Air conditioning",Kitche...	7	1.0	Real Bed	strict	True	...	40.766115	-73.989040	Superb 3BR Apt Located Near Times Square	Hell's Kitchen	6	93.0	https://a0.muscache.com/im/pictures/348a55fe-4...	10019	3.0	3.0
2	7919400	4.976734	Apartment	Entire home/apt	{TV,"Cable TV","Wireless Internet","Air condit...	5	1.0	Real Bed	moderate	True	...	40.808110	-73.943756	The Garden Oasis	Harlem	10	92.0	https://a0.muscache.com/im/pictures/6fae5362-9...	10027	1.0	3.0
3	13418779	6.620073	House	Entire home/apt	{TV,"Cable TV",Internet,"Wireless Internet",Ki...	4	1.0	Real Bed	flexible	True	...	37.772004	-122.431619	Beautiful Flat in the Heart of SF!	Lower Haight	0	NaN	https://a0.muscache.com/im/pictures/72208dad-9...	94117	2.0	2.0
4	3808709	4.744932	Apartment	Entire home/apt	{TV,Internet,"Wireless Internet","Air conditio...	2	1.0	Real Bed	moderate	True	...	38.925627	-77.034596	Great studio in midtown DC	Columbia Heights	4	40.0	NaN	20009	0.0	1.0
5 rows × 29 columns

# Data Exploration Section
## Understanding the Dataset Structure
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
display(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
Dataset Overview:
Shape: (74111, 29)

First 5 rows:
id	log_price	property_type	room_type	amenities	accommodates	bathrooms	bed_type	cancellation_policy	cleaning_fee	...	latitude	longitude	name	neighbourhood	number_of_reviews	review_scores_rating	thumbnail_url	zipcode	bedrooms	beds
0	6901257	5.010635	Apartment	Entire home/apt	{"Wireless Internet","Air conditioning",Kitche...	3	1.0	Real Bed	strict	True	...	40.696524	-73.991617	Beautiful brownstone 1-bedroom	Brooklyn Heights	2	100.0	https://a0.muscache.com/im/pictures/6d7cbbf7-c...	11201	1.0	1.0
1	6304928	5.129899	Apartment	Entire home/apt	{"Wireless Internet","Air conditioning",Kitche...	7	1.0	Real Bed	strict	True	...	40.766115	-73.989040	Superb 3BR Apt Located Near Times Square	Hell's Kitchen	6	93.0	https://a0.muscache.com/im/pictures/348a55fe-4...	10019	3.0	3.0
2	7919400	4.976734	Apartment	Entire home/apt	{TV,"Cable TV","Wireless Internet","Air condit...	5	1.0	Real Bed	moderate	True	...	40.808110	-73.943756	The Garden Oasis	Harlem	10	92.0	https://a0.muscache.com/im/pictures/6fae5362-9...	10027	1.0	3.0
3	13418779	6.620073	House	Entire home/apt	{TV,"Cable TV",Internet,"Wireless Internet",Ki...	4	1.0	Real Bed	flexible	True	...	37.772004	-122.431619	Beautiful Flat in the Heart of SF!	Lower Haight	0	NaN	https://a0.muscache.com/im/pictures/72208dad-9...	94117	2.0	2.0
4	3808709	4.744932	Apartment	Entire home/apt	{TV,Internet,"Wireless Internet","Air conditio...	2	1.0	Real Bed	moderate	True	...	38.925627	-77.034596	Great studio in midtown DC	Columbia Heights	4	40.0	NaN	20009	0.0	1.0
5 rows × 29 columns

Data Types:
id                          int64
log_price                 float64
property_type              object
room_type                  object
amenities                  object
accommodates                int64
bathrooms                 float64
bed_type                   object
cancellation_policy        object
cleaning_fee                 bool
city                       object
description                object
first_review               object
host_has_profile_pic       object
host_identity_verified     object
host_response_rate         object
host_since                 object
instant_bookable           object
last_review                object
latitude                  float64
longitude                 float64
name                       object
neighbourhood              object
number_of_reviews           int64
review_scores_rating      float64
thumbnail_url              object
zipcode                    object
bedrooms                  float64
beds                      float64
dtype: object

Missing Values:
id                            0
log_price                     0
property_type                 0
room_type                     0
amenities                     0
accommodates                  0
bathrooms                   200
bed_type                      0
cancellation_policy           0
cleaning_fee                  0
city                          0
description                   0
first_review              15864
host_has_profile_pic        188
host_identity_verified      188
host_response_rate        18299
host_since                  188
instant_bookable              0
last_review               15827
latitude                      0
longitude                     0
name                          0
neighbourhood              6872
number_of_reviews             0
review_scores_rating      16722
thumbnail_url              8216
zipcode                     968
bedrooms                     91
beds                        131
dtype: int64
# Data Preprocessing
## Handling Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values

df['last_review'].fillna('Never', inplace=True)
df['name'].fillna('Unnamed Listing', inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
Missing Values Before Cleaning:
id                            0
log_price                     0
property_type                 0
room_type                     0
amenities                     0
accommodates                  0
bathrooms                   200
bed_type                      0
cancellation_policy           0
cleaning_fee                  0
city                          0
description                   0
first_review              15864
host_has_profile_pic        188
host_identity_verified      188
host_response_rate        18299
host_since                  188
instant_bookable              0
last_review                   0
latitude                      0
longitude                     0
name                          0
neighbourhood              6872
number_of_reviews             0
review_scores_rating      16722
thumbnail_url              8216
zipcode                     968
bedrooms                     91
beds                        131
dtype: int64

Missing Values After Cleaning:
id                            0
log_price                     0
property_type                 0
room_type                     0
amenities                     0
accommodates                  0
bathrooms                   200
bed_type                      0
cancellation_policy           0
cleaning_fee                  0
city                          0
description                   0
first_review              15864
host_has_profile_pic        188
host_identity_verified      188
host_response_rate        18299
host_since                  188
instant_bookable              0
last_review                   0
latitude                      0
longitude                     0
name                          0
neighbourhood              6872
number_of_reviews             0
review_scores_rating      16722
thumbnail_url              8216
zipcode                     968
bedrooms                     91
beds                        131
dtype: int64
#common words
text = ' '.join(df['name'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Listing Names')
plt.show()
No description has been provided for this image
#Model Development
# Preprocessing for Pipeline
from sklearn.model_selection import train_test_split

# Define Features and Target
X = df.drop(columns=["log_price"])  # Drop the target column from features
y = df["log_price"]  # Define target variable

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=[object]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

# Model Training
model.fit(X_train, y_train)
  Pipeline?i
 preprocessor: ColumnTransformer?
num

 SimpleImputer?

 StandardScaler?
cat

 SimpleImputer?

 OneHotEncoder?

XGBRegressor
#Model Evaluation
y_pred = model.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R²:', r2_score(y_test, y_pred))
RMSE: 0.3929939491412592
MAE: 0.2866367289568788
R²: 0.6993644967251702
## Neighborhood Analysis
plt.figure(figsize=(12, 8))
neighborhood_counts = df['neighbourhood'].value_counts().head(20)
sns.barplot(y=neighborhood_counts.index, x=neighborhood_counts.values)
plt.title('Top 20 Neighborhoods by Listing Count')
plt.xlabel('Number of Listings')
plt.ylabel('Neighborhood')
plt.show()
No description has been provided for this image
## Actual vs Predicted Prices Visualization

import matplotlib.pyplot as plt
import pandas as pd


y_pred = model.predict(X_test)

# Create the DataFrame from y_test and y_pred
df_sample = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True)[:20],
    'Predicted': pd.Series(y_pred).reset_index(drop=True)[:20]
})

# Bar plot
df_sample.plot(kind='bar', figsize=(10, 6), colormap='coolwarm', alpha=0.8)

plt.xlabel('Sample Listings')
plt.ylabel('Log Prices')
plt.title('Bar Plot of Actual vs Predicted Log Prices for Sample Listings')
plt.xticks(rotation=45)
plt.legend(["Actual", "Predicted"])
plt.tight_layout()
plt.show()
No description has been provided for this image
 
 
