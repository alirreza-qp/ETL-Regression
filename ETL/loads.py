def load_csv(data,file_path):
    try:
        data.to_csv(file_path,index=False)
    except:
        print("Something went wrong")
    else:
        print("="*50)
        print('CSV file create successfully')
        print("="*50)

def load_json(data,file_path):
    data.to_json(file_path,index=False)