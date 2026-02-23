"""
Generate colored comparison CSV files from test results
Creates 2 files per row: one with vertical ICD codes and color coding
"""

import pandas as pd
from pathlib import Path
import json

def generate_colored_comparison_csvs(input_csv_path):
    """
    Read the test results CSV and generate colored comparison files
    """
    # Read the main test results
    df = pd.read_csv(input_csv_path)
    
    # Create output directory
    output_dir = Path("testing/outputs/colored_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(df)} test results...")
    
    # Process each row
    for idx, row in df.iterrows():
        filename = row['filename']
        patient_id = row['patient_id']
        
        # Get ICD codes from each test (split by comma)
        test1_codes = [c.strip() for c in str(row['test1']).split(',') if c.strip() and c.strip() != 'nan']
        test2_codes = [c.strip() for c in str(row['test2']).split(',') if c.strip() and c.strip() != 'nan']
        test3_codes = [c.strip() for c in str(row['test3']).split(',') if c.strip() and c.strip() != 'nan']
        test4_codes = [c.strip() for c in str(row['test4']).split(',') if c.strip() and c.strip() != 'nan']
        
        # Find common codes (present in all 4 tests)
        common_codes = set(test1_codes) & set(test2_codes) & set(test3_codes) & set(test4_codes)
        
        # Get all unique codes across all tests
        all_codes = sorted(set(test1_codes + test2_codes + test3_codes + test4_codes))
        
        # Create comparison data
        comparison_data = []
        color_data = []
        
        for code in all_codes:
            row_data = {
                'ICD_Code': code,
                'Test1': code if code in test1_codes else '',
                'Test2': code if code in test2_codes else '',
                'Test3': code if code in test3_codes else '',
                'Test4': code if code in test4_codes else ''
            }
            
            # Determine color (BLUE if in all tests, RED if not)
            color = 'BLUE' if code in common_codes else 'RED'
            color_row = {
                'ICD_Code': '',
                'Test1': color if code in test1_codes else '',
                'Test2': color if code in test2_codes else '',
                'Test3': color if code in test3_codes else '',
                'Test4': color if code in test4_codes else ''
            }
            
            comparison_data.append(row_data)
            color_data.append(color_row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        color_df = pd.DataFrame(color_data)
        
        # Generate safe filename
        safe_filename = Path(filename).stem.replace(' ', '_')[:50]
        
        # File 1: Comparison with codes
        output_file1 = output_dir / f"{idx+1:03d}_{patient_id}_{safe_filename}_comparison.csv"
        comparison_df.to_csv(output_file1, index=False)
        
        # File 2: Color indicators
        output_file2 = output_dir / f"{idx+1:03d}_{patient_id}_{safe_filename}_colors.csv"
        color_df.to_csv(output_file2, index=False)
        
        print(f"  [{idx+1}/{len(df)}] Generated: {output_file1.name}")
        print(f"               and: {output_file2.name}")
    
    print(f"\n✅ Generated {len(df) * 2} CSV files in: {output_dir}")
    print(f"\nLegend:")
    print(f"  BLUE = Code present in all 4 tests (consistent)")
    print(f"  RED  = Code missing in at least 1 test (inconsistent)")
    print(f"  Blank = Code not present in that test")


def main():
    input_csv = Path("testing/outputs/prompt_consistency_test.csv")
    
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        print("   Please run test_prompt_outputs.py first.")
        return
    
    print("=" * 80)
    print("COLORED COMPARISON CSV GENERATOR")
    print("=" * 80)
    
    generate_colored_comparison_csvs(input_csv)


if __name__ == "__main__":
    main()
