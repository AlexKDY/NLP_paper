import os

current_dir = os.path.dirname(__file__)

file_path = os.path.join(current_dir, '..', 'dataset', 'text8')

with open(file_path) as f:
    text = f.read()

print(text[:100])  # first 100 characters

                                                                                                                                                                                                                                                                                                                                                            