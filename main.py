import json

def create_vectors(filename: str) -> json:
    # Holds the aggregated feature vectors for all malware families
    families = dict()
    label_file = open(filename, 'r')

    for line in label_file:
        line = line.strip()
        tokens = line.split(",")

        # Skip the header line
        if (tokens[0] != "sha256"):
            families[tokens[1]] = dict()

            # Open the file corresponding to the hash
            vector_file = open("feature_vectors/" + tokens[0], 'r')
            for feature in vector_file:
                feature = feature.strip()
                # Will only grab the unique features as dict keys
                families[tokens[1]][feature] = 0

            vector_file.close()

    label_file.close()

    families_json = json.dumps(families, indent=4)
    families_file = open("families.json", "w")
    families_file.write(families_json)
    families_file.close()

    return families_json


def main():
    vectors = create_vectors("sha256_family.csv")

if __name__ == "__main__":
    main()