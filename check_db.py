from db import get_all_batches

batches = get_all_batches()
print(f"{len(batches)} batches found")
if batches:
    print(batches[0])
else:
    print("none")