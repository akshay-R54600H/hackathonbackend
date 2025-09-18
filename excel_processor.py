"""
Excel Data Processor for Financial Forecasting

This module extracts financial forecast parameters from Excel files
with month columns (M1, M2, M3, etc.) and various row attributes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class ExcelProcessor:
    """Processes Excel files to extract financial forecast parameters."""
    
    def __init__(self):
        # Define exact patterns to match the specific attributes you mentioned
        # These are case-insensitive and flexible to handle variations
        self.parameter_patterns = {
            'salespeople': [
                r'#\s*of\s*sales\s*people',
                r'number\s*of\s*sales\s*people',
                r'sales\s*people',
                r'salespeople'
            ],
            'deals_per_salesperson': [
                r'#\s*of\s*large\s*customer\s*accounts?\s*they\s*can\s*sign\s*per\s*month\s*/?\s*sales\s*person',
                r'deals?\s*per\s*sales\s*person',
                r'deals?\s*per\s*rep',
                r'accounts?\s*per\s*sales\s*person',
                r'deals?\s*per\s*month\s*per\s*sales\s*person'
            ],
            'large_customer_revenue': [
                r'average\s*revenue\s*per\s*customer\s*\(\$\s*per\s*month\)\s*for\s*large\s*customers',
                r'average\s*revenue\s*per\s*customer.*large',
                r'large\s*customer\s*revenue',
                r'revenue\s*per\s*large\s*customer',
                r'large\s*customer.*revenue.*per\s*month'
            ],
            'marketing_spend': [
                r'digital\s*marketing\s*spend\s*per\s*month\s*\(\$\)',
                r'digital\s*marketing\s*spend',
                r'marketing\s*spend\s*per\s*month',
                r'marketing\s*spend',
                r'ad\s*spend',
                r'advertising\s*spend'
            ],
            'cac': [
                r'average\s*customer\s*acquisition\s*cost\s*\(cac,\s*\$\)',
                r'customer\s*acquisition\s*cost',
                r'average\s*cac',
                r'cac',
                r'acquisition\s*cost',
                r'cost\s*per\s*acquisition'
            ],
            'conversion_rate': [
                r'%\s*conversions?\s*from\s*demo\s*to\s*sign\s*ups?\s*\(%\)',
                r'%?\s*conversions?\s*from\s*demo\s*to\s*sign\s*ups?',
                r'conversion\s*rate',
                r'demo\s*to\s*signup\s*conversion',
                r'conversion\s*percentage',
                r'signup\s*rate'
            ],
            'sme_customer_revenue': [
                r'average\s*revenue\s*per\s*sme\s*customer\s*\(\$\)',
                r'sme\s*customer\s*revenue',
                r'small.*medium.*customer\s*revenue',
                r'average\s*revenue\s*per\s*sme\s*customer',
                r'sme.*revenue.*per\s*customer',
                r'small.*medium.*revenue.*per\s*customer'
            ]
        }
    
    def extract_month_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract month columns (M1, M2, M3, etc.) from the DataFrame."""
        month_columns = []
        
        # First, try to find columns with M1, M2, M3, etc. format
        for col in df.columns:
            col_str = str(col).strip().upper()
            if re.match(r'^M\d+$', col_str) or re.match(r'^MONTH\s*\d+$', col_str):
                month_columns.append(col)
        
        # If no M1, M2, M3 columns found, check if we have numeric columns (1, 2, 3, etc.)
        if not month_columns:
            for col in df.columns:
                col_str = str(col).strip()
                # Check if column is numeric (1, 2, 3, etc.)
                if col_str.isdigit() and 1 <= int(col_str) <= 12:
                    month_columns.append(col)
        
        # Sort by numeric value
        if month_columns:
            try:
                return sorted(month_columns, key=lambda x: int(re.findall(r'\d+', str(x))[0]))
            except:
                return month_columns
        
        return month_columns
    
    def normalize_row_name(self, row_name: str) -> str:
        """Normalize row names for pattern matching."""
        return re.sub(r'[^\w\s]', ' ', str(row_name).lower()).strip()
    
    def find_matching_row(self, df: pd.DataFrame, patterns: List[str]) -> Optional[Tuple[int, str]]:
        """Find the first row that matches any of the given patterns."""
        for idx, row_name in enumerate(df.index):
            normalized_name = self.normalize_row_name(row_name)
            for pattern in patterns:
                # Try exact match first, then flexible match
                if re.search(pattern, normalized_name, re.IGNORECASE):
                    return idx, str(row_name)
                # Also try with more flexible matching (remove special chars)
                flexible_name = re.sub(r'[^\w\s]', ' ', normalized_name)
                if re.search(pattern, flexible_name, re.IGNORECASE):
                    return idx, str(row_name)
        return None
    
    def extract_numeric_values(self, series: pd.Series) -> List[float]:
        """Extract numeric values from a pandas Series, handling various formats."""
        values = []
        
        for val in series:
            if pd.isna(val):
                values.append(0.0)
                continue
            
            # Convert to string and clean
            val_str = str(val).strip()
            original_val = val_str
            
            # Check if the original value contains a % symbol
            has_percentage_symbol = '%' in original_val
            
            # Remove common prefixes/suffixes
            val_str = re.sub(r'^[$£€¥]', '', val_str)  # Remove currency symbols
            val_str = re.sub(r'[%]$', '', val_str)     # Remove percentage sign
            val_str = re.sub(r'[,]', '', val_str)      # Remove commas
            
            # Try to convert to float
            try:
                numeric_value = float(val_str)
                
                # If the original value had a % symbol, convert to decimal
                if has_percentage_symbol:
                    # Convert percentage to decimal (e.g., 5000% -> 50.0, 50% -> 0.5)
                    # This is because 5000% should be interpreted as 50.0 for the user's use case
                    numeric_value = numeric_value / 100.0
                
                values.append(numeric_value)
                
            except ValueError:
                values.append(0.0)
        
        return values
    
    def calculate_average(self, values: List[float]) -> float:
        """Calculate average of non-zero values."""
        non_zero_values = [v for v in values if v != 0]
        if not non_zero_values:
            return 0.0
        return sum(non_zero_values) / len(non_zero_values)
    
    def detect_salespeople_growth_pattern(self, values: List[float]) -> Tuple[float, float]:
        """
        Detect if salespeople values show a growth pattern and calculate initial + monthly increment.
        
        Args:
            values: List of salespeople values across months
            
        Returns:
            Tuple of (initial_salespeople, monthly_increment)
        """
        if len(values) < 2:
            return values[0] if values else 0, 0
        
        # Check if values are increasing
        is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        
        if is_increasing:
            # Calculate monthly increment (average difference)
            increments = [values[i+1] - values[i] for i in range(len(values)-1)]
            monthly_increment = sum(increments) / len(increments) if increments else 0
            initial_salespeople = values[0]
            return initial_salespeople, monthly_increment
        else:
            # If not increasing, use average as initial, no monthly increment
            return sum(values) / len(values), 0

    def process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an Excel file and extract financial forecast parameters.
        This acts as a complete replacement for Gemini, producing the same JSON structure.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing extracted parameters in Gemini format
        """
        try:
            # Read the Excel file with dtype as object to preserve original format
            df = pd.read_excel(file_path, index_col=0, dtype=str)
            
            # Extract month columns
            month_columns = self.extract_month_columns(df)
            
            if not month_columns:
                raise ValueError("No month columns (M1, M2, M3, etc.) found in the Excel file")
            
            print(f"🔍 Found month columns: {month_columns}")
            print(f"📊 Found rows: {list(df.index)}")
            print(f"🎯 Looking for these specific attributes:")
            for param_name, patterns in self.parameter_patterns.items():
                print(f"   • {param_name}: {patterns[0]} (and variations)")
            
            # Initialize result dictionary
            result = {
                'months': len(month_columns),
                'extracted_data': {},
                'raw_data': {},
                'debug_info': {
                    'total_rows': len(df),
                    'month_columns_found': month_columns,
                    'all_row_names': list(df.index)
                }
            }
            
            # Extract each parameter with detailed debugging
            for param_name, patterns in self.parameter_patterns.items():
                match_result = self.find_matching_row(df, patterns)
                
                if match_result:
                    row_idx, row_name = match_result
                    print(f"✅ Found {param_name} in row: '{row_name}'")
                    
                    # Extract values for this row across month columns
                    row_data = df.iloc[row_idx][month_columns]
                    values = self.extract_numeric_values(row_data)
                    average_value = self.calculate_average(values)
                    
                    # Debug: show the actual values being processed
                    print(f"  📋 Raw values: {list(row_data)}")
                    print(f"  🔢 Extracted values: {values}")
                    print(f"  📊 Calculated average: {average_value}")
                    
                    # Special handling for salespeople to detect growth patterns
                    if param_name == 'salespeople':
                        initial_sp, monthly_inc = self.detect_salespeople_growth_pattern(values)
                        result['extracted_data']['salespeople'] = initial_sp
                        result['extracted_data']['salespeople_added_per_month'] = monthly_inc
                        print(f"  📈 Initial salespeople: {initial_sp}")
                        print(f"  📈 Monthly increment: {monthly_inc}")
                    else:
                        result['extracted_data'][param_name] = average_value
                        print(f"  📊 Average {param_name}: {average_value}")
                    
                    result['raw_data'][param_name] = {
                        'row_name': row_name,
                        'values': values,
                        'month_columns': month_columns,
                        'average': average_value
                    }
                else:
                    print(f"❌ Could not find {param_name} in the Excel file")
                    print(f"   Searched patterns: {patterns}")
                    result['extracted_data'][param_name] = None
                    result['debug_info'][f'missing_{param_name}'] = {
                        'searched_patterns': patterns,
                        'available_rows': list(df.index)
                    }
            
            # Add comprehensive debugging information
            result['debug_info']['extraction_summary'] = {
                'total_parameters': len(self.parameter_patterns),
                'found_parameters': len([k for k, v in result['extracted_data'].items() if v is not None]),
                'missing_parameters': [k for k, v in result['extracted_data'].items() if v is None]
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing Excel file: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    def convert_to_assumptions(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert extracted data to forecast assumptions format.
        This mimics the exact JSON structure that Gemini would return.
        
        Args:
            extracted_data: Data extracted from Excel file
            
        Returns:
            Dictionary in the exact format expected by the forecast system
        """
        data = extracted_data.get('extracted_data', {})
        months = extracted_data.get('months', 12)
        
        # Create assumptions in the exact format Gemini would return
        assumptions = {
            'months': months,
            'initial_salespeople': data.get('salespeople'),
            'salespeople_added_per_month': data.get('salespeople_added_per_month'),
            'deals_per_salesperson': data.get('deals_per_salesperson'),
            'large_customer_revenue_per_month': data.get('large_customer_revenue'),
            'marketing_spend_per_month': data.get('marketing_spend'),
            'average_cac': data.get('cac'),
            'conversion_rate': data.get('conversion_rate'),
            'sme_customer_revenue_per_month': data.get('sme_customer_revenue'),
            'overrides': []
        }
        
        # Replace None values with MISSING_* format (same as Gemini)
        missing_fields = []
        for key, value in assumptions.items():
            if value is None and key != 'overrides':
                assumptions[key] = f"MISSING_{key}"
                missing_fields.append(f"MISSING_{key}")
        
        # Add debugging information
        assumptions['_debug'] = {
            'missing_fields': missing_fields,
            'extracted_data': extracted_data,
            'source': 'excel_processor'
        }
        
        return assumptions


def process_excel_file(file_path: str) -> Dict[str, Any]:
    """
    Process an Excel file and return the exact same JSON structure as Gemini.
    This function acts as a complete replacement for Gemini API calls.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary in the exact format that Gemini would return
    """
    processor = ExcelProcessor()
    extracted_data = processor.process_excel_file(file_path)
    assumptions = processor.convert_to_assumptions(extracted_data)
    
    # Return in the exact same format as Gemini
    return {
        'assumptions': assumptions,
        'extracted_data': extracted_data,  # For debugging
        'source': 'excel_processor',
        'message': 'Excel file processed successfully'
    }


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python excel_processor.py <excel_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        result = process_excel_file(file_path)
        print("\n=== EXTRACTED DATA ===")
        print(result['extracted_data'])
        print("\n=== ASSUMPTIONS ===")
        print(result['assumptions'])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)