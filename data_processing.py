import pandas as pd


def load_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        data = pd.read_json(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format!")
    return data

def preprocess_data(data):
    # Handling missing values
    data = data.fillna(data.median(numeric_only=True)) 
    # Removing duplicates
    data = data.drop_duplicates()
    
     # Identifying categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    
   # Encoding categorical variables
    for col in categorical_columns:
        if data[col].nunique() < 10:  # Optional: Only encode columns with a small number of unique values
            data = pd.get_dummies(data, columns=[col], drop_first=True)  # drop_first=True to avoid multicollinearity
        else:
            data[col] = data[col].astype('category').cat.codes  # Label encoding for high cardinality
    
    
     # Removing outliers using the IQR method
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
   
    
    return data