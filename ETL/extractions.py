import pandas as pd

def extract_from_csv(file_path):
    try:
        pd.read_csv(file_path)
    except:
        print("'Something went wrong'")
    else:
        print("="*50)
        print('CSV file read successfully')
        print("="*50)
        return pd.read_csv(file_path)
     
     
     
def extract_from_json(file_path):
    return pd.read_json(file_path)


# def extract_from_csv(file_path):
#     return pd.read_csv(file_path)


def extract_from_excel(file_path):
    return pd.read_excel(file_path)


def extract_from_xml(file_path):
    return pd.read_xml(file_path)