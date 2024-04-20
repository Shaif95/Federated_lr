import json
import matplotlib.pyplot as plt

# Load accuracy data from file
with open("client_accuracy.json", "r") as f:
    acc_list = json.load(f)

print(acc_list)

# Plot the accuracy data
plt.figure(figsize=(10, 6))
plt.plot(acc_list, label='Training Accuracy', marker='o')
plt.title('Training Accuracy Over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
