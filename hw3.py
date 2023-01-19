# Q1a)A-Correct, B-Wrong, C-Correct, D-Correct
# Q1b)0.95

from sklearn.metrics import normalized_mutual_info_score

predicted_labels = Y.argmax(axis=1)
nmi_score = normalized_mutual_info_score(labels, predicted_labels)
nmi_score_rounded = round(nmi_score, 2)

print("NMI score:", nmi_score_rounded)
