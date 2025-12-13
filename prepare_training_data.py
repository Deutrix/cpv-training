import json
import csv
import glob
import os

def process_contracts(contracts_dir, writer):
    json_files = glob.glob(os.path.join(contracts_dir, "*.json"))
    print(f"Found {len(json_files)} files in contracts directory.")
    
    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data is expected to be a list of dictionaries
                if isinstance(data, list):
                    for item in data:
                        tender_name = item.get("TenderName")
                        cpv = item.get("Cpv")
                        
                        if tender_name and cpv:
                            # Clean up strings if necessary (e.g., remove newlines)
                            tender_name = tender_name.strip()
                            cpv = cpv.strip()
                            writer.writerow([tender_name, cpv])
                            count += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    print(f"Extracted {count} records from contracts.")

def process_postupci(postupci_dir, writer):
    json_files = glob.glob(os.path.join(postupci_dir, "*.json"))
    print(f"Found {len(json_files)} files in postupci directory.")
    
    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data is expected to be a list of dictionaries
                if isinstance(data, list):
                    for item in data:
                        name = item.get("Name")
                        cpv_extended = item.get("CPVExtended")
                        
                        if name and cpv_extended:
                            name = name.strip()
                            cpv_extended = cpv_extended.strip()
                            writer.writerow([name, cpv_extended])
                            count += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    print(f"Extracted {count} records from postupci.")

def main():
    base_dir = r"c:\Users\boris.AORUS\Desktop\cpv-decoder\dataset"
    contracts_dir = os.path.join(base_dir, "contracts")
    postupci_dir = os.path.join(base_dir, "postupci")
    output_file = r"c:\Users\boris.AORUS\Desktop\cpv-decoder\training_data.csv"

    print("Starting data extraction...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['input', 'output'])
        
        if os.path.exists(contracts_dir):
            process_contracts(contracts_dir, writer)
        else:
            print(f"Contracts directory not found: {contracts_dir}")
            
        if os.path.exists(postupci_dir):
            process_postupci(postupci_dir, writer)
        else:
            print(f"Postupci directory not found: {postupci_dir}")

    print(f"Data extraction complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
