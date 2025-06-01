import matplotlib.pyplot as plt
from Bio import SeqIO

gbff_file = "data/E_coli_K12.gbff"

cds_lengths = []
for record in SeqIO.parse(gbff_file, "genbank"):
    for feature in record.features:
        if feature.type == "CDS":
            start = int(feature.location.start)
            end = int(feature.location.end)
            cds_lengths.append(end - start)


plt.figure(figsize=(8, 5))
plt.hist(cds_lengths, bins=50)
plt.title("Histogram of CDS Lengths in E. coli K-12")
plt.xlabel("CDS Length (bp)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
