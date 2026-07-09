import os
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)

def main():
    os.makedirs('data_zenodo', exist_ok=True)
    base_url = 'https://zenodo.org/records/7648117/files/'
    files = ['X_test.csv', 'X_train.csv', 'y_test.csv', 'y_train.csv']
    
    for filename in files:
        url = f"{base_url}{filename}?download=1"
        dest = os.path.join('data_zenodo', filename)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"{filename} already exists, skipping.")
    print("All files downloaded successfully.")

if __name__ == '__main__':
    main()
