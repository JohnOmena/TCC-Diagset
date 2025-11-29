from label_schemes import (
    collapse_to_S1,
    collapse_to_S2,
    collapse_to_S3,
    collapse_to_S4,
)

# rótulos S5 de exemplo
# correspondem a: [BG, T, N, A, R1, R2, R3, R4, R5]
labels_s5 = list(range(9))

print("Labels S5:", labels_s5)
print("S1:", collapse_to_S1(labels_s5).tolist())
print("S2:", collapse_to_S2(labels_s5).tolist())
print("S3:", collapse_to_S3(labels_s5).tolist())
print("S4:", collapse_to_S4(labels_s5).tolist())
