import re

def generate_bigrams(text):
    words=re.findall(r'\b\w+\b', text.lower())
    return [(words[i], words[i+1]) for i in range(len(words)-1)
    ]

text= "Hello, world! Apple releases: a new iPhone."
text="Hello"
print(generate_bigrams(text))

