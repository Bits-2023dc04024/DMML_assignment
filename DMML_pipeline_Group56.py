



import os
import logging
import pandas as pd
import json
import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# Prefect Flow for Pipeline Orchestration
from prefect import flow, task

# Global configuration
CONFIG = {
    "parent_dir": "C:/Users/ajays/Desktop/BITS 2 sem/DMML_Pipeline/",
    "raw_data_dir": "raw_data",                 # Will create dataset-specific subfolders (telco, bank)
    "clean_data_dir": "clean_data",             # Dataset-specific cleaned data folders: clean_data/telco, clean_data/bank, etc.
    "transformed_data_dir": "transformed_data", # Dataset-specific transformed data folders
    "logs_dir": "logs",
    "raw_db_name": "bronze",   
    "cleansed_db_name": "silver",   
    "transformed_db_name": "gold",                       # SQLite DBs stored in dataset-specific subfolders
    "model_save_path": "model",                 # Model files in dataset-specific subfolders
    "feature_store_path": "feature_store",      # Feature store metadata in dataset-specific subfolders
    "model_report_path": "model_report",        # Model reports in dataset-specific subfolders
    "version_metadata_file": "version_metadata",# Global folder for version metadata JSON
    "validation_report_path": "validation_report",
    # For Telco the target column is "Churn"; for Bank it is "churn"
    "target_column_telco": "Churn",
    "target_column_bank": "churn",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "timestamp_format": "%Y%m%d_%H%M%S",
    # Kaggle dataset configuration
    "kaggle_dataset_telco": "blastchar/telco-customer-churn",
    "kaggle_csv_telco": "Telco-Customer-Churn.csv",
    "kaggle_dataset_bank": "gauravtopre/bank-customer-churn-dataset",
    "kaggle_csv_bank": "BankChurn.csv"
}

@task
def save_versioned_data(df, folder, filename_prefix, change_log):
    """
    Saves the dataset to a versioned CSV inside the specified folder (typically a 'versions/' folder).
    Logs metadata in a global JSON file.
    """
    version_metadata_folder = os.path.join(CONFIG["parent_dir"], CONFIG["version_metadata_file"])
    file_name = "version_metadata.json"
    version_metadata_file = os.path.join(version_metadata_folder, file_name)
    
    # Ensure the version folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Create a timestamp-based version tag
    timestamp = datetime.now().strftime(CONFIG["timestamp_format"])
    version_tag = f"v{timestamp}"
    filename = f"{filename_prefix}_{version_tag}.csv"
    filepath = os.path.join(folder, filename)
    
    # Save the CSV file
    df.to_csv(filepath, index=False)
    
    # Load or initialize version metadata JSON
    if os.path.exists(version_metadata_file):
        with open(version_metadata_file, "r") as f:
            version_data = json.load(f)
    else:
        version_data = {}
    
    version_data[filename] = {
        "version": version_tag,
        "timestamp": datetime.now().isoformat(),
        "source": folder,
        "change_log": change_log
    }
    
    os.makedirs(version_metadata_folder, exist_ok=True)
    with open(version_metadata_file, "w") as f:
        json.dump(version_data, f, indent=4)
    
    logging.info(f"Saved versioned file: {filepath}")
    return filepath

@task
def create_paths(CONFIG):   
    """
    Creates required directories based on configuration.
    """
    DIR_KEYS = [
        "raw_data_dir", "clean_data_dir", "logs_dir", "transformed_data_dir", 
        "validation_report_path", "version_metadata_file", "raw_db_name", 
        "cleansed_db_name", "transformed_db_name", 
        "feature_store_path", "model_report_path", "model_save_path"
    ]
    
    for key in DIR_KEYS:
        CONFIG[key] = os.path.join(CONFIG["parent_dir"], CONFIG[key])
        os.makedirs(CONFIG[key], exist_ok=True)
    
    # Configure logging
    logs_dir = CONFIG["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, "pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def store_raw_data_in_db(df, dataset_name):
    """
    Stores raw ingested data in an SQLite database.
    """
    db_folder = os.path.join(CONFIG["raw_db_name"], dataset_name)
    os.makedirs(db_folder, exist_ok=True)
    db_path = os.path.join(db_folder, f"raw_data.db")
    
    conn = sqlite3.connect(db_path)
    df.to_sql("raw_data", conn, if_exists="replace", index=False)
    conn.close()
    
    logging.info(f"Raw data stored in SQLite: {db_path}")
    return df


def store_cleansed_data_in_db(df, dataset_name):
    """
    Stores cleansed data in an SQLite database.
    """
    db_folder = os.path.join(CONFIG["cleansed_db_name"], dataset_name)
    os.makedirs(db_folder, exist_ok=True)
    db_path = os.path.join(db_folder, f"cleansed_data.db")
    
    conn = sqlite3.connect(db_path)
    df.to_sql("cleansed_data", conn, if_exists="replace", index=False)
    conn.close()
    
    logging.info(f"Cleansed data stored in SQLite: {db_path}")
    return df


@task
def data_validation(df, dataset_name):
    """
    Validates the dataset for missing values, duplicates, and data types.
    Saves a dataset-specific validation report as CSV.
    """
    logging.info("Starting data validation...")
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    
    # Create a dataset-specific subfolder inside the validation report path
    validation_folder = os.path.join(CONFIG["validation_report_path"], dataset_name)
    os.makedirs(validation_folder, exist_ok=True)
    
    # Use a dataset-specific file name for the report
    validation_file_name = f"validation_report_{dataset_name}.csv"
    validation_report_path = os.path.join(validation_folder, validation_file_name)
    
    pd.DataFrame([report]).to_csv(validation_report_path, index=False)
    logging.info(f"Data validation report saved: {validation_report_path}")
    
    if df.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in dataset.")
    if df.duplicated().sum() > 0:
        logging.warning("Duplicate records detected in dataset.")
    
    logging.info("Data validation completed.")
    return df


@task
def data_ingestion_kaggle(dataset_name):
    """
    Ingests the specified dataset (telco or bank) from Kaggle.
    Always overwrites the master CSV file in the dataset folder.
    Saves a versioned copy in a 'versions/' subfolder.
    """
    logging.info(f"Starting data ingestion for {dataset_name} dataset from Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Determine dataset-specific values for raw data
    if dataset_name.lower() == "telco":
        dataset_slug = CONFIG["kaggle_dataset_telco"]
        csv_filename = CONFIG["kaggle_csv_telco"]
        dataset_folder = os.path.join(CONFIG["raw_data_dir"], "telco")
    elif dataset_name.lower() == "bank":
        dataset_slug = CONFIG["kaggle_dataset_bank"]
        csv_filename = CONFIG["kaggle_csv_bank"]
        dataset_folder = os.path.join(CONFIG["raw_data_dir"], "bank")
    else:
        logging.error("Invalid dataset name provided.")
        return None

    os.makedirs(dataset_folder, exist_ok=True)

    # Remove old master CSV if exists to force overwrite
    master_csv_path = os.path.join(dataset_folder, csv_filename)
    if os.path.exists(master_csv_path):
        os.remove(master_csv_path)
        logging.info(f"Removed old master CSV: {master_csv_path}")

    # Download & unzip dataset files from Kaggle
    logging.info(f"Downloading from Kaggle: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=dataset_folder, unzip=True)

    # Expect exactly one CSV file after download
    csv_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".csv")]
    if len(csv_files) == 0:
        logging.error("No CSV files found after Kaggle download.")
        return None
    elif len(csv_files) > 1:
        logging.error(f"Multiple CSV files found ({csv_files}). Expected exactly one for {dataset_name}.")
        return None

    # Rename the downloaded CSV to the master CSV name from CONFIG
    downloaded_csv = csv_files[0]
    downloaded_csv_path = os.path.join(dataset_folder, downloaded_csv)
    if downloaded_csv != csv_filename:
        os.rename(downloaded_csv_path, master_csv_path)
        logging.info(f"Renamed {downloaded_csv} to {csv_filename}")
    else:
        logging.info(f"Master CSV already named correctly: {csv_filename}")

    # Read the master CSV with dataset-specific handling
    if dataset_name.lower() == "telco":
        df = pd.read_csv(master_csv_path, dtype={'TotalCharges': 'str'})
    else:  # bank dataset
        df = pd.read_csv(master_csv_path)
    
    # Validate data
    df = data_validation(df,dataset_name)
    #save in database
    store_raw_data_in_db(df, dataset_name)
    # Save a versioned copy in the 'versions/' subfolder (for raw data)
    version_folder = os.path.join(dataset_folder, "versions")
    os.makedirs(version_folder, exist_ok=True)
    save_versioned_data(
        df,
        version_folder,
        f"raw_data_{dataset_name}",
        f"Original raw data ingested for {dataset_name}"
    )
    logging.info(f"Data ingestion for {dataset_name} dataset completed successfully.")
    return df

@task
def data_preparation(df, dataset_name):
    """
    Cleans and prepares the dataset with dataset-specific transformations.
    Saves a versioned cleaned CSV in a dataset-specific folder.
    Also creates dataset-specific diagrams with labels, stored in separate folders.
    """
    logging.info("Starting data preparation...")
    
    if dataset_name.lower() == "telco":
        # Telco-specific cleaning
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        cat_cols = ["gender", "InternetService", "MultipleLines", "OnlineSecurity", 
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                    "StreamingMovies", "Contract", "PaymentMethod"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        target_col = "Churn"
    elif dataset_name.lower() == "bank":
        # Bank-specific cleaning
        if "customer_id" in df.columns:
            df.drop(columns=["customer_id"], inplace=True)
        if "churn" in df.columns:
            df["churn"] = df["churn"].astype(int)
        cat_cols = ["country", "gender"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        target_col = "churn"

    store_cleansed_data_in_db(df, dataset_name)

    # Save cleaned data in a dataset-specific subfolder
    cleaned_folder = os.path.join(CONFIG["clean_data_dir"], dataset_name)
    os.makedirs(cleaned_folder, exist_ok=True)
    save_versioned_data(df, cleaned_folder, "cleaned_data", f"Cleaned data with feature engineering for {dataset_name}")
    
    # Create a dataset-specific diagrams folder
    diagrams_folder = os.path.join(CONFIG["logs_dir"], "diagrams", dataset_name)
    os.makedirs(diagrams_folder, exist_ok=True)
    
    # Visualizations with dataset labels in titles
    plt.figure(figsize=(8, 5))
    if target_col in df.columns:
        sns.countplot(x=target_col, data=df)
        plt.title(f"Churn Distribution ({dataset_name.upper()})")
        plt.savefig(os.path.join(diagrams_folder, f"churn_distribution_{dataset_name}.png"))
        plt.close()
    
    plt.figure(figsize=(12, 6))
    df.hist(figsize=(12, 6), bins=20)
    plt.suptitle(f"Feature Distributions ({dataset_name.upper()})")
    plt.savefig(os.path.join(diagrams_folder, f"feature_distributions_{dataset_name}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    numeric_corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Matrix ({dataset_name.upper()})")
    plt.savefig(os.path.join(diagrams_folder, f"correlation_matrix_{dataset_name}.png"))
    plt.close()
    
    logging.info("Data preparation completed successfully with visualizations.")
    return df


@task
def data_transformation(df, dataset_name):
    """
    Performs feature engineering and scaling.
    Saves a versioned transformed CSV in a dataset-specific folder.
    Also creates a SQLite database for the transformed data.
    """
    logging.info("Starting data transformation...")
    
    # Convert tenure to years if available.
    if "tenure" in df.columns:
        df["tenure_years"] = df["tenure"] / 12.0
    
    # Additional feature engineering for the Telco dataset.
    if dataset_name.lower() == "telco":
        # Convert TotalCharges to numeric if not already done.
        if df["TotalCharges"].dtype == 'object':
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        # Average monthly charge: TotalCharges divided by tenure (avoid division by zero)
        if "tenure" in df.columns and "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
            df["avg_monthly_charge"] = df.apply(
                lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else row["MonthlyCharges"],
                axis=1
            )
        # Create a categorical feature by binning MonthlyCharges
        if "MonthlyCharges" in df.columns:
            df["charge_category"] = pd.cut(
                df["MonthlyCharges"],
                bins=[0, 35, 70, 100],
                labels=["low", "medium", "high"]
            )
            # Encode the charge category as a numeric feature
            df["charge_category_encoded"] = df["charge_category"].map({"low": 0, "medium": 1, "high": 2})
        # Interaction feature: product of MonthlyCharges and tenure
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["charge_tenure_interaction"] = df["MonthlyCharges"] * df["tenure"]
        # Indicator for long-term customers: tenure greater than 12 months
        if "tenure" in df.columns:
            df["is_long_term"] = (df["tenure"] > 12).astype(int)
        # Additional feature: Binning tenure into categories
        df["tenure_category"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["new", "established", "loyal", "veteran"]
        )
    
    # Additional feature engineering for the Bank dataset.
    if dataset_name.lower() == "bank":
        # Updated column names are lowercase: balance and estimated_salary.
        if "balance" in df.columns and "estimated_salary" in df.columns:
            df["salary_balance_ratio"] = df["balance"] / (df["estimated_salary"] + 1e-5)
        # Additional feature: Binning age into groups.
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[18, 30, 45, 60, 100],
                labels=["young", "adult", "middle_aged", "senior"]
            )
    
    # Apply StandardScaler to all numerical features (including newly engineered ones).
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save transformed data in a dataset-specific folder.
    transformed_folder = os.path.join(CONFIG["transformed_data_dir"], dataset_name)
    os.makedirs(transformed_folder, exist_ok=True)
    save_versioned_data(
        df,
        transformed_folder,
        "transformed_data",
        f"Feature engineered and scaled data for {dataset_name}"
    )
    
    # Create SQLite DB for transformed data (dataset-specific).
    db_folder = os.path.join(CONFIG["transformed_db_name"], dataset_name)
    os.makedirs(db_folder, exist_ok=True)
    db_file_name = f"{dataset_name}_churn.db"
    full_db_path = os.path.join(db_folder, db_file_name)
    conn = sqlite3.connect(full_db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS churn_data")
    
    # Clean column names to remove spaces or hyphens.
    df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
    column_types = []
    for column, dtype in df.dtypes.items():
        col_type = "REAL" if "int" in str(dtype) or "float" in str(dtype) else "TEXT"
        column_types.append(f"[{column}] {col_type}")
    
    schema_query = f"CREATE TABLE churn_data ({', '.join(column_types)});"
    cursor.execute(schema_query)
    conn.commit()
    df.to_sql("churn_data", conn, if_exists="replace", index=False)
    conn.close()
    
    logging.info(f"Transformed data stored in SQLite at {full_db_path}.")
    return df



@task
def create_feature_store(df, dataset_name):
    """
    Creates a JSON feature store metadata file based on the transformed dataset.
    Saves the feature store JSON in a dataset-specific subfolder.
    """
    logging.info("Creating feature store...")
    feature_metadata = {}
    for column in df.columns:
        feature_metadata[column] = {
            "description": f"Feature derived from transformed data ({column})",
            "source": f"transformed_data_{dataset_name}.csv",
            "version": "1.0"
        }
    feature_store_folder = os.path.join(CONFIG["feature_store_path"], dataset_name)
    os.makedirs(feature_store_folder, exist_ok=True)
    feature_store_file = os.path.join(feature_store_folder, f"feature_store_{dataset_name}.json")
    with open(feature_store_file, "w") as f:
        json.dump(feature_metadata, f, indent=4)
    logging.info(f"Feature store metadata saved: {feature_store_file}")


@task
def model_training(dataset_name):
    """
    Trains a RandomForest model using the transformed data from the SQLite database.
    Saves the model and a versioned report in dataset-specific subfolders.
    """
    logging.info("Starting model training...")
    
    # Load transformed data from SQLite DB (dataset-specific)
    db_folder = os.path.join(CONFIG["transformed_db_name"], dataset_name)
    db_file_name = f"{dataset_name}_churn.db"
    full_db_path = os.path.join(db_folder, db_file_name)
    conn = sqlite3.connect(full_db_path)
    df = pd.read_sql_query("SELECT * FROM churn_data", conn)
    conn.close()
    
    # Load feature store metadata (dataset-specific)
    feature_store_folder = os.path.join(CONFIG["feature_store_path"], dataset_name)
    feature_store_file = os.path.join(feature_store_folder, f"feature_store_{dataset_name}.json")
    try:
        with open(feature_store_file, "r") as f:
            metadata = json.load(f)
        feature_list = list(metadata.keys())
    except FileNotFoundError:
        logging.warning("Feature store file not found. Using all columns except target as features.")
        feature_list = [col for col in df.columns if col != ("Churn" if dataset_name.lower() == "telco" else "churn")]
    
    target_col = "Churn" if dataset_name.lower() == "telco" else "churn"
    if target_col not in df.columns:
        logging.error(f"Target column {target_col} not found in {dataset_name} dataset.")
        return None
    
    X = df[feature_list]
    y = df[target_col].astype(int)
    
    # Ensure all features are numeric by one-hot encoding non-numeric columns
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
    model = RandomForestClassifier(n_estimators=CONFIG["n_estimators"], random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    timestamp = datetime.now().strftime(CONFIG["timestamp_format"])
    model_version = f"v{timestamp}"
    model_filename = f"{dataset_name}_churn_model_{model_version}.pkl"
    model_folder = os.path.join(CONFIG["model_save_path"], dataset_name)
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    report_filename = f"{dataset_name}_model_report_{model_version}.json"
    report_folder = os.path.join(CONFIG["model_report_path"], dataset_name)
    os.makedirs(report_folder, exist_ok=True)
    report_path = os.path.join(report_folder, report_filename)
    model_report = {
        "version": model_version,
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "timestamp": timestamp
    }
    with open(report_path, "w") as f:
        json.dump(model_report, f, indent=4)
    
    # Update global model versions metadata
    version_metadata_file = os.path.join(CONFIG["model_report_path"], "model_versions.json")
    if os.path.exists(version_metadata_file):
        with open(version_metadata_file, "r") as f:
            model_versions = json.load(f)
    else:
        model_versions = {}
    
    model_versions[model_version] = {
        "model_path": model_path,
        "report_path": report_path,
        "accuracy": accuracy,
        "timestamp": timestamp,
        "dataset": dataset_name
    }
    
    with open(version_metadata_file, "w") as f:
        json.dump(model_versions, f, indent=4)
    
    logging.info(f"Model training for {dataset_name} completed. Version: {model_version}, Accuracy: {accuracy:.4f}")
    return model_version















from prefect import flow
import logging

@flow(name="DMML_pipeline_Group-56",log_prints=True,flow_run_name = "Ajay")

def run_pipeline():
    print("Starting the ML pipeline...")
    create_paths(CONFIG)
    
    # Process Telco first, then Bank
    for ds in ["telco", "bank"]:
        print(f"\n--- Processing {ds.upper()} dataset ---")
        df = data_ingestion_kaggle(ds)
        if df is None or df.empty:
            logging.error(f"Data ingestion failed for {ds}. Skipping this dataset.")
            continue
        
        df = data_validation(df, ds)
        df_clean = data_preparation(df, ds)
        df_transformed = data_transformation(df_clean, ds)
        create_feature_store(df_transformed, ds)
        model_training(ds)
        logging.info(f"Pipeline execution completed for {ds} dataset.")
        print(f"Pipeline completed for {ds.upper()} dataset.")



# if __name__ == "__main__":
#     run_pipeline()






