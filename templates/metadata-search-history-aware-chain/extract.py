import csv, json

def extract_list():
    csvfile = "test.csv"

    header_to_extract = "mediator areas of practice"

    values = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                practice_list = text.split('|')

                for practice in practice_list:
                    new_practice = practice.strip()

                    if not new_practice in values and not new_practice.isdigit():
                        values.append(new_practice)

    jsonfile_path = "list.json"

    with open(jsonfile_path, 'w') as file:
        json.dump(values, file, indent=4)

    return values
