import json
import argparse


def read_data_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return {str(i): json.loads(line) for i, line in enumerate(lines)}

def write_data_to_file(file_name, data):
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent=4))

def write_data_from_list_dict_to_file(data, file_name):
    # Add index to each dictionary and write to file
    indexed_data = {str(i): item for i, item in enumerate(data)}
    with open(file_name, 'w') as f:
        f.write(json.dumps(indexed_data, indent=4))



# # Read data from file and add index
# indexed_data = read_data_from_file(file_name)
#
# print("indexed_data", indexed_data)
# # Write indexed data back to file
# write_data_to_file(file_name, indexed_data)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, required=True,
                        help='txt file to change the format and add index to')
    parser.add_argument('--seed', required=True, help='for selecting the appropriate file', type=int)

    args = parser.parse_args()

    # Read data from file and add index
    indexed_data = read_data_from_file(args.file_name+"_"+str(args.seed))

    # print("indexed_data", indexed_data)
    # Write indexed data back to file
    write_data_to_file(args.file_name+"_"+str(args.seed), indexed_data)