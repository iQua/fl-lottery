import os, sys

def main():
    path = sys.argv[1]
    for round_id in range(1, len(os.listdir(path))+1):
        acc_path = os.path.join(path, str(round_id))
        if os.path.isdir(acc_path):
            if os.path.exists(os.path.join(acc_path, "accuracy.json")):
                with open(os.path.join(acc_path, "accuracy.json")) as acc_fd:
                    line = acc_fd.readline()
                    print(round_id, line)


if __name__ == "__main__":
    main()
