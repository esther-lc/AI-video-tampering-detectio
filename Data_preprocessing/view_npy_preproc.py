# view_npy.py
import numpy as np

# Change this to your actual .npy file path
file_path = r"H:\project_data\Data_Preprocessing\feature_extraction\preprocessed\010_Dup_061_070_CCTV Customer Footage(E).npy"  # or whatever your file is, e.g. X.npy or one preprocessed file

try:
    data = np.load(file_path)
    print("File loaded successfully!")
    print("Shape of the array:", data.shape)          # e.g. (number_of_pairs, 256) for features
    print("Data type:", data.dtype)
    print("\nFirst few rows (first 5 rows, first 10 columns):")
    print(data[:5, :10])  # Adjust slicing as needed

    # If it's 1D or 2D small array, print more
    if data.ndim <= 2 and data.size < 10000:
        print("\nFull content:")
        print(data)
    else:
        print("\nArray is large — showing summary stats instead:")
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())
        print("Std dev:", data.std())

except Exception as e:
    print("Error loading file:", e)