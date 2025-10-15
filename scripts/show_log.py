# with open("./logs/log_2025-09-27_14-29-20.log", "r") as file:
#     for line in file:
#         if "Duration" in line:
#             print(line.strip())


from collections import deque

# Replace 'input.txt' with your file name
filename = "./logs/log_2025-09-27_14-29-20.log"

# Number of lines before/after to print
N = 14

# Use a deque to store previous lines
prev_lines = deque(maxlen=N)

with open(filename, "r") as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    # Check if the line contains the target string
    if "Wall-Clock Time" in line:
        # Print previous N lines
        for prev in prev_lines:
            print(prev.strip())
        # Print the current line
        print(line.strip())
        # Print next N lines (make sure we don't go out of range)
        for next_line in lines[i + 1 : i + 1 + N]:
            print(next_line.strip())
        print("-" * 40)  # Separator for readability

    # Add the current line to deque for future reference
    prev_lines.append(line)
