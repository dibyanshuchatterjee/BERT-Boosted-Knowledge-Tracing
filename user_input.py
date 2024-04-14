

def get_input():
    # Required input
    file_path = input("Enter a prepared KT dataset's path (csv or similar file): ")
    while not file_path:
        print("This field is required. Please enter a value.")
        file_path = input("Enter a prepared KT dataset's path (csv or similar file): ")

    target_col = input("Enter name of target column: ")
    while not target_col:
        print("This field is required. Please enter a value.")
        target_col = input("Enter name of target column: ")

    return file_path, target_col

