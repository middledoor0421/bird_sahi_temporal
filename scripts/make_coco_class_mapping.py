import json

def main():
    mapping = {str(i): i + 1 for i in range(80)}
    with open("class_mapping_coco_plus1.json", "w") as f:
        json.dump(mapping, f, indent=2)

if __name__ == "__main__":
    main()
