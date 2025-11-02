from operator import index

characters = "0123456789abcdefghijklmnopqrstuvwxyz ,.'-"

def hash_fun(data) -> int: # Converts a string to a 3 digit in 000 - 999
    hash_key = 0
    if len(data) >= 2:
        for c in data[:3]:
            hash_key += characters.index(c) + hash_key * characters.index(c)
            hash_key = hash_key % 1000
    return hash_key


words = ["help", "welcome", "heal", "rude", "resist", "consistent", "ray", "abby", "baby", "bout", "cat", "dog", "zebra"]

for word in words:
    print(f"{word}: {hash_fun(word)}")

