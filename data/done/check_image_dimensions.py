def get_shape(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        if num_lines == 0:
            print("Empty file.")
            return

        # Assuming the first line contains data and the values are separated by spaces
        sample_line = lines[0].strip().split()
        num_values = len(sample_line)

        print(f"Number of lines: {num_lines}")
        print(f"Number of values per line: {num_values}")

if __name__ == "__main__":
    filename = "cifar_image.txt"
    get_shape(filename)

