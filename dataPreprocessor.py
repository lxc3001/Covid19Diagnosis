import json
import pandas as pd


class Preprocessor:
    def __init__(self, file_path, output_path=None, save=False):
        self.data = pd.read_csv(file_path)
        self.filtered_data = self.filter_data()
        self.remaining_data = self.get_remaining_data()
        self.num_unique_patient = self.filtered_data['patientid'].nunique() if not self.filtered_data.empty else 0

        if save and output_path:
            self.save_data(output_path)

    @staticmethod
    def is_image_file(filename):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        return isinstance(filename, str) and filename.lower().endswith(image_extensions)
               
    def filter_data(self): 
        # Remove rows where a clinical_notes or offset is either NaN or an empty string
        data_filtered = self.data[['patientid', 'offset', 'sex', 'age', 'finding', 'view', 'date', 'folder', 'filename', 'clinical_notes', 'other_notes']]
        data_filtered = data_filtered[data_filtered['clinical_notes'].notna() & (data_filtered['clinical_notes'] != '')]
        data_filtered = data_filtered[data_filtered['offset'].notna() & (data_filtered['offset'] != '')]

        # Keeep only the rows where filename is an image file
        data_filtered = data_filtered[data_filtered['filename'].apply(self.is_image_file)]

        # Filter patients has different offsets
        patient_offset_counts = data_filtered.groupby('patientid')['offset'].nunique()
        valid_patients = patient_offset_counts[patient_offset_counts > 1].index
        return data_filtered[data_filtered['patientid'].isin(valid_patients)]
    
    def get_remaining_data(self):
        remaining_data = self.data.loc[~self.data.index.isin(self.filtered_data.index)]
        remaining_data = remaining_data[remaining_data['clinical_notes'].notna() & (remaining_data['clinical_notes'] != '')]
        remaining_data = remaining_data[remaining_data['filename'].apply(self.is_image_file)]
        return remaining_data.reset_index(drop=True)  
    
    def save_data(self, output_path):
        # self.filtered_data.to_csv(output_path, index=False)
        # print(f"Filtered data saved to: {output_path}")
        filtered_path = output_path
        remaining_path = output_path.replace('filtered_data.csv', 'remaining_data.csv')

        self.filtered_data.to_csv(filtered_path, index=False)
        self.remaining_data.to_csv(remaining_path, index=False)

        print(f"Filtered data saved to: {filtered_path}")
        print(f"Remaining data saved to: {remaining_path}")

    
if __name__ == "__main__":
    file_path = "./covid-chestxray-dataset-master/data/metadata.csv"
    save_path = './covid-chestxray-dataset-master/data/filtered_data.csv'
    
    preprocessor = Preprocessor(file_path, output_path=save_path, save=True)
    print(f"Number of unique patients after filtering: {preprocessor.num_unique_patient}")
    print(len(preprocessor.filtered_data))
    print(len(preprocessor.remaining_data))

