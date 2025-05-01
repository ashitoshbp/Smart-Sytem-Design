import pandas as pd
import os
import json
from datetime import datetime

class DataPreprocessor:
    def __init__(self, excel_path):
        """
        Initialize the DataPreprocessor with the path to the Excel file.
        
        Args:
            excel_path (str): Path to the Excel file containing incident data
        """
        self.excel_path = excel_path
        self.data = None
        
    def load_data(self):
        """
        Load data from the Excel file.
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        print(f"Loading data from {self.excel_path}")
        self.data = pd.read_excel(self.excel_path)
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def clean_data(self):
        """
        Clean and preprocess the data.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.data is None:
            self.load_data()
            
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Convert column names to lowercase and replace spaces with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        df = df.fillna({
            'description': 'No description provided',
            'location': 'Unknown location',
            'status': 'Unknown'
        })
        
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                print(f"Could not convert {col} to datetime")
        
        # Calculate incident duration if open and close dates exist
        if 'open_date' in df.columns and 'close_date' in df.columns:
            df['duration_hours'] = (df['close_date'] - df['open_date']).dt.total_seconds() / 3600
            
        # Create a text field that combines relevant information for embedding
        text_columns = ['incident_type', 'description', 'location', 'status']
        text_columns = [col for col in text_columns if col in df.columns]
        
        df['combined_text'] = df[text_columns].apply(
            lambda row: ' '.join(str(val) for val in row if pd.notna(val)), 
            axis=1
        )
        
        self.data = df
        print(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def save_processed_data(self, output_path):
        """
        Save the processed data to a CSV file.
        
        Args:
            output_path (str): Path to save the processed data
        """
        if self.data is None:
            raise ValueError("No data to save. Please load and clean data first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        # Also save a JSON version for easier consumption by the RAG system
        json_path = output_path.replace('.csv', '.json')
        
        # Convert datetime columns to strings for JSON serialization
        json_df = self.data.copy()
        for col in json_df.columns:
            if pd.api.types.is_datetime64_any_dtype(json_df[col]):
                json_df[col] = json_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
        # Save to JSON
        json_df.to_json(json_path, orient='records', date_format='iso')
        print(f"JSON version saved to {json_path}")
        
    def get_data_stats(self):
        """
        Get basic statistics about the data.
        
        Returns:
            dict: Dictionary containing data statistics
        """
        if self.data is None:
            raise ValueError("No data available. Please load data first.")
            
        stats = {
            "total_records": len(self.data),
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
        }
        
        # Get incident type distribution if it exists
        if 'incident_type' in self.data.columns:
            stats["incident_type_distribution"] = self.data['incident_type'].value_counts().to_dict()
            
        # Get status distribution if it exists
        if 'status' in self.data.columns:
            stats["status_distribution"] = self.data['status'].value_counts().to_dict()
            
        return stats

# Example usage
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent directory (backend) and then to the data directory
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # Define paths
    excel_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "Incident_Report (1).xlsx")
    output_path = os.path.join(data_dir, "processed_incidents.csv")
    
    # Create an instance of DataPreprocessor
    preprocessor = DataPreprocessor(excel_path)
    
    # Load and clean the data
    preprocessor.load_data()
    preprocessor.clean_data()
    
    # Save the processed data
    preprocessor.save_processed_data(output_path)
    
    # Get and print data statistics
    stats = preprocessor.get_data_stats()
    print("\nData Statistics:")
    print(json.dumps(stats, indent=2))
