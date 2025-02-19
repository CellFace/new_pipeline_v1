# coding=utf-8

# Standard Library Imports
import os

# Third Party Imports
import pandas as pd
import ast
import re

# Local Application/Library Specific Imports
from base.settings import LoggerMixin

class PostDataProcessor(LoggerMixin):
    
    def __init__(self):
        super().__init__()

    def load_csv_file(self, csv_file_path: str):
        """
            Loads, processes, and saves an enhanced version of a CSV file containing cell count and aggregation data.

            Args:
                csv_file_path (str): Path to the input CSV file containing measurement and aggregation details.

            Process:
                - Reads relevant columns from the CSV file.
                - Renames columns for clarity.
                - Extracts patient IDs based on the measurement directory.
                - Parses aggregate details and computes additional count metrics.
                - Calculates various percentages related to cell types, single cell counts, and aggregation metrics.
                - Ensures meaningful aggregation-to-total ratios.
                - Generates a unique filename if an output file with the same name already exists.
                - Saves the processed data as a new CSV file.

            Returns:
                None (saves the processed DataFrame to a new CSV file).
        """
        df = pd.read_csv(csv_file_path, usecols=[
            'Measurement Directory', 'Number of frames', 'RBC Count', 'WBC Count', 'PLT Count', 
            'PLT-PLT AGG Count', 'WBC-PLT AGG Count', 'WBC-WBC AGG Count', 'Aggregate Image Info'
        ])
        df.columns = [
            'Measurement Directory', 'Number of frames', 'Total RBC Count', 'Total WBC Count', 'Total PLT Count', 
            'Total PLT-PLT Agg Count', 'Total WBC-PLT Agg Count', 'Total WBC-WBC Agg Count', 'Agg Details'
        ]
        
        df['patient'] = df['Measurement Directory'].apply(lambda x: self.extract_patient_id(x))

        additional_counts = df['Agg Details'].apply(self.parse_agg_details)
        df = pd.concat([df, additional_counts], axis=1)
                
        df['Total Cell Count'] = df['Total RBC Count'] + df['Total WBC Count'] + df['Total PLT Count']
        df['Single WBC Count'] = df['Total WBC Count'] - df['wbc_in_agg_count']
        df['Single PLT Count'] = df['Total PLT Count'] - df['plt_in_agg_count']
        
        df['total_rbc_percentage'] = (df['Total RBC Count'] / df['Total Cell Count']) * 100
        df['total_wbc_percentage'] = (df['Total WBC Count'] / df['Total Cell Count']) * 100
        df['total_plt_percentage'] = (df['Total PLT Count'] / df['Total Cell Count']) * 100
        df['single_plt_percentage'] = (df['Single PLT Count'] / df['Total Cell Count']) * 100
        df['single_wbc_percentage'] = (df['Single WBC Count'] / df['Total Cell Count']) * 100
        df['plt_agg_percentage'] = (df['Total PLT-PLT Agg Count'] / df['Total Cell Count']) * 100
        df['lp_agg_percentage'] = (df['Total WBC-PLT Agg Count'] / df['Total Cell Count']) * 100
        df['ll_agg_percentage'] = (df['Total WBC-WBC Agg Count'] / df['Total Cell Count']) * 100
        df['plt_agg_percentage_to_plt'] = (df['Total PLT-PLT Agg Count'] / (df['Total PLT Count'])) * 100
        df['lp_agg_percentage_to_plt'] = (df['Total WBC-PLT Agg Count'] / df['Total PLT Count']) * 100
        df['ll_agg_percentage_to_wbc'] = (df['Total WBC-WBC Agg Count'] / df['Total WBC Count']) * 100
        # ----------
        df['lp_agg_1l_percentage'] = (df['lp_agg_1l_count'] / df['Total Cell Count']) * 100
        df['lp_agg_2l_percentage'] = (df['lp_agg_2l_count'] / df['Total Cell Count']) * 100
        df['lp_agg_3l_percentage'] = (df['lp_agg_3l_count'] / df['Total Cell Count']) * 100
        df['lp_agg_4l_percentage'] = (df['lp_agg_4l_count'] / df['Total Cell Count']) * 100
        df['lp_agg_5plusl_percentage'] = (df['lp_agg_5plusl_count'] / df['Total Cell Count']) * 100
        df['lp_agg_1p_percentage'] = (df['lp_agg_1p_count'] / df['Total Cell Count']) * 100
        df['lp_agg_2p_percentage'] = (df['lp_agg_2p_count'] / df['Total Cell Count']) * 100
        df['lp_agg_3p_percentage'] = (df['lp_agg_3p_count'] / df['Total Cell Count']) * 100
        df['lp_agg_4p_percentage'] = (df['lp_agg_4p_count'] / df['Total Cell Count']) * 100
        df['lp_agg_5plusp_percentage'] = (df['lp_agg_5plusp_count'] / df['Total Cell Count']) * 100
        df['plt_agg_2p_percentage'] = (df['plt_agg_2p_count'] / df['Total Cell Count']) * 100
        df['plt_agg_3p_percentage'] = (df['plt_agg_3p_count'] / df['Total Cell Count']) * 100
        df['plt_agg_4p_percentage'] = (df['plt_agg_4p_count'] / df['Total Cell Count']) * 100
        df['plt_agg_5plusp_percentage'] = (df['plt_agg_5plusp_count'] / df['Total Cell Count']) * 100
        df['ll_agg_2l_percentage'] = (df['ll_agg_2l_count'] / df['Total Cell Count']) * 100
        df['ll_agg_3l_percentage'] = (df['ll_agg_3l_count'] / df['Total Cell Count']) * 100
        df['ll_agg_4l_percentage'] = (df['ll_agg_4l_count'] / df['Total Cell Count']) * 100
        df['ll_agg_5plusl_percentage'] = (df['ll_agg_5plusl_count'] / df['Total Cell Count']) * 100
        # ----------
        df['lp_agg_1l_percentage_to_lp_agg_percentage'] = (df['lp_agg_1l_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_2l_percentage_to_lp_agg_percentage'] = (df['lp_agg_2l_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_3l_percentage_to_lp_agg_percentage'] = (df['lp_agg_3l_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_4l_percentage_to_lp_agg_percentage'] = (df['lp_agg_4l_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_5plusl_percentage_to_lp_agg_percentage'] = (df['lp_agg_5plusl_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_1p_percentage_to_lp_agg_percentage'] = (df['lp_agg_1p_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_2p_percentage_to_lp_agg_percentage'] = (df['lp_agg_2p_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_3p_percentage_to_lp_agg_percentage'] = (df['lp_agg_3p_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_4p_percentage_to_lp_agg_percentage'] = (df['lp_agg_4p_percentage'] / df['lp_agg_percentage']) * 100
        df['lp_agg_5plusp_percentage_to_lp_agg_percentage'] = (df['lp_agg_5plusp_percentage'] / df['lp_agg_percentage']) * 100
        df['plt_agg_2p_percentage_to_plt_agg_percentage'] = (df['plt_agg_2p_percentage'] / df['plt_agg_percentage']) * 100
        df['plt_agg_3p_percentage_to_plt_agg_percentage'] = (df['plt_agg_3p_percentage'] / df['plt_agg_percentage']) * 100
        df['plt_agg_4p_percentage_to_plt_agg_percentage'] = (df['plt_agg_4p_percentage'] / df['plt_agg_percentage']) * 100
        df['plt_agg_5plusp_percentage_to_plt_agg_percentage'] = (df['plt_agg_5plusp_percentage'] / df['plt_agg_percentage']) * 100
        df['ll_agg_2l_percentage_to_ll_agg_percentage'] = (df['ll_agg_2l_percentage'] / df['ll_agg_percentage']) * 100
        df['ll_agg_3l_percentage_to_ll_agg_percentage'] = (df['ll_agg_3l_percentage'] / df['ll_agg_percentage']) * 100
        df['ll_agg_4l_percentage_to_ll_agg_percentage'] = (df['ll_agg_4l_percentage'] / df['ll_agg_percentage']) * 100
        df['ll_agg_5plusl_percentage_to_ll_agg_percentage'] = (df['ll_agg_5plusl_percentage'] / df['ll_agg_percentage']) * 100
        # ----------
        df['plt_agg_2p_percentage_to_plt'] = (df['plt_agg_2p_count'] / (df['Total PLT Count'])) * 100
        df['plt_agg_3p_percentage_to_plt'] = (df['plt_agg_3p_count'] / (df['Total PLT Count'])) * 100
        df['plt_agg_4p_percentage_to_plt'] = (df['plt_agg_4p_count'] / (df['Total PLT Count'])) * 100
        df['plt_agg_5plusp_percentage_to_plt'] = (df['plt_agg_5plusp_count'] / (df['Total PLT Count'])) * 100
        df['lp_agg_1l_percentage_to_plt'] = (df['lp_agg_1l_count'] / df['Total PLT Count']) * 100
        df['lp_agg_2l_percentage_to_plt'] = (df['lp_agg_2l_count'] / df['Total PLT Count']) * 100
        df['lp_agg_3l_percentage_to_plt'] = (df['lp_agg_3l_count'] / df['Total PLT Count']) * 100
        df['lp_agg_4l_percentage_to_plt'] = (df['lp_agg_4l_count'] / df['Total PLT Count']) * 100
        df['lp_agg_5plusl_percentage_to_plt'] = (df['lp_agg_5plusl_count'] / df['Total PLT Count']) * 100
        df['lp_agg_1p_percentage_to_plt'] = (df['lp_agg_1p_count'] / df['Total PLT Count']) * 100
        df['lp_agg_2p_percentage_to_plt'] = (df['lp_agg_2p_count'] / df['Total PLT Count']) * 100
        df['lp_agg_3p_percentage_to_plt'] = (df['lp_agg_3p_count'] / df['Total PLT Count']) * 100
        df['lp_agg_4p_percentage_to_plt'] = (df['lp_agg_4p_count'] / df['Total PLT Count']) * 100
        df['lp_agg_5plusp_percentage_to_plt'] = (df['lp_agg_5plusp_count'] / df['Total PLT Count']) * 100
        df['ll_agg_2l_percentage_to_wbc'] = (df['ll_agg_2l_count'] / df['Total WBC Count']) * 100
        df['ll_agg_3l_percentage_to_wbc'] = (df['ll_agg_3l_count'] / df['Total WBC Count']) * 100
        df['ll_agg_4l_percentage_to_wbc'] = (df['ll_agg_4l_count'] / df['Total WBC Count']) * 100
        df['ll_agg_5plusl_percentage_to_wbc'] = (df['ll_agg_5plusl_count'] / df['Total WBC Count']) * 100
        # ----------
        df['plt_in_agg_percentage'] = (df['plt_in_agg_count'] / df['Total PLT Count']) * 100
        df['plt_agg_ratio'] = df['plt_in_agg_count'] / df['Single PLT Count']
        df['wbc_in_agg_percentage'] = (df['wbc_in_agg_count'] / df['Total WBC Count']) * 100
        df['wbc_agg_ratio'] = df['wbc_in_agg_count'] / df['Single WBC Count']

        columns_to_include = [
            'Measurement Directory', 'patient', 'Total Cell Count', 'Total RBC Count', 'Total WBC Count', 'Total PLT Count', 
            'Single WBC Count', 'Single PLT Count', 'Total PLT-PLT Agg Count', 'Total WBC-PLT Agg Count', 'Total WBC-WBC Agg Count', 
            'plt_in_agg_count', 'wbc_in_agg_count', 'lp_agg_1l_count', 'lp_agg_2l_count', 'lp_agg_3l_count', 'lp_agg_4l_count', 'lp_agg_5plusl_count', 
            'lp_agg_1p_count', 'lp_agg_2p_count', 'lp_agg_3p_count', 'lp_agg_4p_count', 'lp_agg_5plusp_count',
            'plt_agg_2p_count', 'plt_agg_3p_count', 'plt_agg_4p_count', 'plt_agg_5plusp_count',
            'll_agg_2l_count', 'll_agg_3l_count', 'll_agg_4l_count', 'll_agg_5plusl_count',
            'total_rbc_percentage', 'total_wbc_percentage', 'total_plt_percentage', 'single_plt_percentage', 'single_wbc_percentage',
            'plt_agg_percentage', 'lp_agg_percentage', 'll_agg_percentage', 'plt_agg_percentage_to_plt', 'lp_agg_percentage_to_plt', 'll_agg_percentage_to_wbc',
            'lp_agg_1l_percentage', 'lp_agg_2l_percentage', 'lp_agg_3l_percentage', 'lp_agg_4l_percentage', 'lp_agg_5plusl_percentage',
            'lp_agg_1p_percentage', 'lp_agg_2p_percentage', 'lp_agg_3p_percentage', 'lp_agg_4p_percentage', 'lp_agg_5plusp_percentage',
            'plt_agg_2p_percentage', 'plt_agg_3p_percentage', 'plt_agg_4p_percentage', 'plt_agg_5plusp_percentage',
            'll_agg_2l_percentage', 'll_agg_3l_percentage', 'll_agg_4l_percentage', 'll_agg_5plusl_percentage',
            'lp_agg_1l_percentage_to_lp_agg_percentage', 'lp_agg_2l_percentage_to_lp_agg_percentage', 'lp_agg_3l_percentage_to_lp_agg_percentage', 'lp_agg_4l_percentage_to_lp_agg_percentage', 'lp_agg_5plusl_percentage_to_lp_agg_percentage',
            'lp_agg_1p_percentage_to_lp_agg_percentage', 'lp_agg_2p_percentage_to_lp_agg_percentage', 'lp_agg_3p_percentage_to_lp_agg_percentage', 'lp_agg_4p_percentage_to_lp_agg_percentage', 'lp_agg_5plusp_percentage_to_lp_agg_percentage',
            'plt_agg_2p_percentage_to_plt_agg_percentage', 'plt_agg_3p_percentage_to_plt_agg_percentage', 'plt_agg_4p_percentage_to_plt_agg_percentage', 'plt_agg_5plusp_percentage_to_plt_agg_percentage',
            'll_agg_2l_percentage_to_ll_agg_percentage', 'll_agg_3l_percentage_to_ll_agg_percentage', 'll_agg_4l_percentage_to_ll_agg_percentage', 'll_agg_5plusl_percentage_to_ll_agg_percentage',
            'plt_agg_2p_percentage_to_plt', 'plt_agg_3p_percentage_to_plt', 'plt_agg_4p_percentage_to_plt', 'plt_agg_5plusp_percentage_to_plt', 
            'lp_agg_1l_percentage_to_plt', 'lp_agg_2l_percentage_to_plt', 'lp_agg_3l_percentage_to_plt', 'lp_agg_4l_percentage_to_plt', 'lp_agg_5plusl_percentage_to_plt',
            'lp_agg_1p_percentage_to_plt', 'lp_agg_2p_percentage_to_plt', 'lp_agg_3p_percentage_to_plt', 'lp_agg_4p_percentage_to_plt', 'lp_agg_5plusp_percentage_to_plt',
            'll_agg_2l_percentage_to_wbc', 'll_agg_3l_percentage_to_wbc', 'll_agg_4l_percentage_to_wbc', 'll_agg_5plusl_percentage_to_wbc',
            'plt_in_agg_percentage', 'plt_agg_ratio', 'wbc_in_agg_percentage', 'wbc_agg_ratio'
        ]


        # Generate a new file name if the file already exists
        base, ext = os.path.splitext(csv_file_path)
        counter = 1
        new_csv_file_path = f"{base}_{counter}{ext}"
        while os.path.exists(new_csv_file_path):
            counter += 1
            new_csv_file_path = f"{base}_{counter}{ext}"

        df[columns_to_include].to_csv(new_csv_file_path, index=False)

    def extract_patient_id(self, text):
        match = re.search(r'\b(?:CFP|CFE|PC)\d{3}-\d\b', text)
        return match.group(0) if match else None

    def parse_agg_details(self, details):
        """
            Parses aggregate details from a given string representation and extracts cell count metrics.

            Args:
                details (str or NaN): String representation of aggregate details, or NaN if missing.

            Process:
                - Initializes a dictionary to store aggregate counts.
                - Checks if `details` is not NaN and attempts to parse it using `ast.literal_eval()`.
                - Iterates over detected aggregates and extracts white blood cell (WBC) and platelet (PLT) counts.
                - Categorizes aggregates based on different conditions and increments corresponding count keys.
                - Logs any parsing errors for debugging.
                
            Returns:
                pd.Series: A Pandas Series containing counts of different aggregate types:
                    - `plt_in_agg_count`: Total platelets involved in aggregates.
                    - `wbc_in_agg_count`: Total white blood cells involved in aggregates.
                    - `lp_agg_Xl_count`: Aggregates with X white blood cells and at least one platelet.
                    - `lp_agg_Xp_count`: Aggregates with at least one white blood cell and X platelets.
                    - `plt_agg_Xp_count`: Pure platelet aggregates of size X.
                    - `ll_agg_Xl_count`: Pure white blood cell aggregates of size X.
        """
        counts = {
            'wbc_count_wbcagg':0,
            'plt_count_pltagg':0,
            'plt_in_agg_count': 0,
            'wbc_in_agg_count': 0,
            'lp_agg_1l_count': 0,
            'lp_agg_2l_count': 0,
            'lp_agg_3l_count': 0,
            'lp_agg_4l_count': 0,
            'lp_agg_5plusl_count': 0,
            'lp_agg_1p_count': 0,
            'lp_agg_2p_count': 0,
            'lp_agg_3p_count': 0,
            'lp_agg_4p_count': 0,
            'lp_agg_5plusp_count': 0,
            'plt_agg_2p_count': 0,
            'plt_agg_3p_count': 0,
            'plt_agg_4p_count': 0,
            'plt_agg_5plusp_count': 0,
            'll_agg_2l_count': 0,
            'll_agg_3l_count': 0,
            'll_agg_4l_count': 0,
            'll_agg_5plusl_count': 0,
        }
        if pd.notna(details):
            try:
                agg_list = ast.literal_eval(details)
                for item in agg_list:
                    for subitem in item:
                        if isinstance(subitem, dict):
                            counts['plt_in_agg_count'] += subitem.get('p', 0)
                            counts['wbc_in_agg_count'] += subitem.get('w', 0)
                            wbc_count_wbcagg = subitem.get('w', 0)
                            plt_count_pltagg = subitem.get('p', 0)
                            if wbc_count_wbcagg == 1 and plt_count_pltagg > 0:
                                counts['lp_agg_1l_count'] += 1
                            elif wbc_count_wbcagg == 2 and plt_count_pltagg > 0:
                                counts['lp_agg_2l_count'] += 1
                            elif wbc_count_wbcagg == 3 and plt_count_pltagg > 0:
                                counts['lp_agg_3l_count'] += 1
                            elif wbc_count_wbcagg == 4 and plt_count_pltagg > 0:
                                counts['lp_agg_4l_count'] += 1
                            elif wbc_count_wbcagg > 4 and plt_count_pltagg > 0:
                                counts['lp_agg_5plusl_count'] += 1
                            if wbc_count_wbcagg > 0 and plt_count_pltagg == 1:
                                counts['lp_agg_1p_count'] += 1
                            elif wbc_count_wbcagg > 0 and plt_count_pltagg == 2:
                                counts['lp_agg_2p_count'] += 1
                            elif wbc_count_wbcagg > 0 and plt_count_pltagg == 3:
                                counts['lp_agg_3p_count'] += 1
                            elif wbc_count_wbcagg > 0 and plt_count_pltagg == 4:
                                counts['lp_agg_4p_count'] += 1
                            elif wbc_count_wbcagg > 0 and plt_count_pltagg > 4:
                                counts['lp_agg_5plusp_count'] += 1
                            if  wbc_count_wbcagg == 0 and plt_count_pltagg == 2:
                                counts['plt_agg_2p_count'] += 1 
                            elif wbc_count_wbcagg == 0 and plt_count_pltagg == 3:
                                counts['plt_agg_3p_count'] += 1 
                            elif wbc_count_wbcagg == 0 and plt_count_pltagg == 4:
                                counts['plt_agg_4p_count'] += 1
                            elif wbc_count_wbcagg == 0 and plt_count_pltagg > 4:
                                counts['plt_agg_5plusp_count'] += 1
                            if wbc_count_wbcagg == 2 and plt_count_pltagg == 0:
                                counts['ll_agg_2l_count'] += 1
                            elif wbc_count_wbcagg == 3 and plt_count_pltagg == 0:
                                counts['ll_agg_3l_count'] += 1
                            elif wbc_count_wbcagg == 4 and plt_count_pltagg == 0:
                                counts['ll_agg_4l_count'] += 1
                            elif wbc_count_wbcagg > 4 and plt_count_pltagg == 0:
                                counts['ll_agg_5plusl_count'] += 1

            except Exception as e:
                self.logger.error(f"Failed to parse: {details} due to error: {e}")
        return pd.Series(counts)

