def create_vectors(filename: str) -> dict:
    families = dict() # Holds the aggregated feature vectors for all malware families
    label_file = open(filename, 'r')

    for line in label_file:
        line = line.strip()
        tokens = line.split(",")

        if (tokens[0] != "sha256"): # Skip the header line
            families[tokens[1]] = dict()

            vector_file = open("feature_vectors/" + tokens[0], 'r')
            for feature in vector_file:
                feature = feature.strip()
                # Will only grab the unique features as dict keys
                families[tokens[1]][feature] = 0

            vector_file.close()

    label_file.close()

def main():
    create_vectors("sha256_family.csv")

if __name__ == "__main__":
    main()